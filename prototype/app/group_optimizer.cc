#include "app/group_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>

#include "app/group_config.h"
#include "app/global.h"

namespace {

// Constants mirroring the Python reference.
constexpr int    kMaxGroups = 4;
const int        kSegBlocks = static_cast<int>(kSegmentBlocks);
constexpr int    kPageBytes = 4096;
constexpr double kPhysLogicalFactor = 1.1; // physical = logical * 1.1
constexpr double kOverheadPerGroupSeg = 0.5;
constexpr double kUserTrafficThreshold = 0.0005;
constexpr double kEpsTraffic = 1e-12;

struct PLogRecord {
  uint64_t idx;
  int cat;       // predicted_group + 1
  int realGroup;
  int64_t realInterval;
};

using Hist = std::unordered_map<int, int64_t>; // seg_idx -> count

struct EvalResult {
  bool ok = false;
  double waf = 0.0;
  int K = 0;
  std::vector<int> mList;
  std::vector<int> BIR; // per group [1..K]
  std::vector<double> weights;     // group weights (Free->Gi cold split)
  std::vector<double> coldDirect;  // Free->Gi cold
  std::vector<double> coldTotalIn; // recursive cold inflow
  std::vector<double> totalIn;     // total inflow (cold + hot routed to G1)
};

// Category->group mapping shared across placement and optimizer.
static std::vector<int> g_mList;
static std::mutex g_mapMutex;
static std::vector<int> g_firstGcTarget;

double Clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

double LambertW0(double z) {
  // Simple Newton iteration for principal branch.
  double w = (z < 1.0) ? z : std::log(z + 1e-9);
  for (int i = 0; i < 50; ++i) {
    double ew = std::exp(w);
    double wew = w * ew;
    double f = wew - z;
    double denom = ew * (w + 1.0);
    if (std::fabs(denom) < 1e-15) break;
    double wNew = w - f / denom;
    if (std::fabs(wNew - w) < 1e-12) break;
    w = wNew;
  }
  return w;
}

double FifoLastGroupVr(double op) {
  op = Clamp01(op);
  op = std::max(1e-6, std::min(0.999999, op));
  double x = 1.0 / op;
  double z = -x * std::exp(-x);
  double w = LambertW0(z);
  double waf = x / (x + w);
  double vr = 1.0 - (1.0 / waf);
  if (vr >= 1.0) vr = 0.999;
  if (vr <= 0.05) vr = 0.8;
  return vr;
}

bool ReadPLog(const std::string &path, std::vector<PLogRecord> *out) {
  std::ifstream fin(path);
  if (!fin.is_open()) return false;
  std::string line;
  bool first = true;
  while (std::getline(fin, line)) {
    if (first) { first = false; continue; } // skip header
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string tok;
    std::vector<std::string> cols;
    while (std::getline(ss, tok, ',')) cols.push_back(tok);
    if (cols.size() < 4) continue;
    PLogRecord rec;
    rec.idx = std::strtoull(cols[0].c_str(), nullptr, 10);
    int pred = std::atoi(cols[1].c_str());
    rec.cat = pred + 1;
    rec.realGroup = std::atoi(cols[2].c_str());
    rec.realInterval = std::strtoll(cols[3].c_str(), nullptr, 10);
    out->push_back(rec);
  }
  return !out->empty();
}

bool ReadHot(const std::string &path, int *hotSize, double *hotInvalid, double *pHot) {
  std::ifstream fin(path);
  if (!fin.is_open()) return false;
  double sz = 0.0, inv = 0.0, traffic = 0.0;
  fin >> sz >> inv >> traffic;
  if (!fin) return false;
  *hotSize = static_cast<int>(sz);
  *hotInvalid = Clamp01(inv);
  *pHot = Clamp01(traffic);
  return true;
}

bool ReadBir(const std::string &path, int *birHot, std::unordered_map<int,int> *birByCat) {
  std::ifstream fin(path);
  if (!fin.is_open()) return false;
  std::string line;
  std::vector<int> vals;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    double v = std::atof(line.c_str());
    if (v < 0) break;
    vals.push_back(static_cast<int>(v));
  }
  if (vals.empty()) return false;
  *birHot = vals[0];
  for (size_t i = 1; i < vals.size(); ++i) {
    (*birByCat)[static_cast<int>(i)] = vals[i];
  }
  return true;
}

void BuildPLog(const std::vector<PLogRecord> &recs,
               std::unordered_map<int, Hist> *plog,
               std::unordered_map<int, std::vector<int>> *catSegs,
               int *maxCat) {
  *maxCat = 0;
  for (const auto &r : recs) {
    int sidx = static_cast<int>(r.realInterval / kSegBlocks);
    (*plog)[r.cat][sidx] += 1;
    (*catSegs)[r.cat].push_back(sidx);
    if (r.cat > *maxCat) *maxCat = r.cat;
  }
}

std::vector<std::vector<int>> GenerateMLists(int maxCat, int maxGroups=kMaxGroups) {
  std::vector<std::vector<int>> out;
  if (maxCat <= 0) return out;
  const bool forceHighCatsToLast = (maxCat >= 7); // ensure 7+ stay in final group
  std::vector<int> inner;
  for (int i = 1; i < maxCat; ++i) inner.push_back(i);
  for (int K = 1; K <= maxGroups; ++K) {
    if (K == 1) {
      out.push_back({0, maxCat});
      continue;
    }
    // choose K-1 boundaries from inner in order
    std::vector<int> idx(K - 1);
    std::function<void(int,int)> dfs = [&](int start, int depth) {
      if (depth == K - 1) {
        std::vector<int> m = {0};
        for (int d = 0; d < K-1; ++d) m.push_back(inner[idx[d]]);
        m.push_back(maxCat);
        // Enforce: categories >=7 must fall into the last group.
        // Last group starts at (m[K-1]+1); keep that start <=6 when maxCat>=7.
        if (forceHighCatsToLast && K >= 2 && m[K-1] >= 7) return;
        out.push_back(std::move(m));
        return;
      }
      for (int pos = start; pos < static_cast<int>(inner.size()); ++pos) {
        idx[depth] = pos;
        dfs(pos + 1, depth + 1);
      }
    };
    dfs(0, 0);
  }
  return out;
}

std::pair<std::unordered_map<int,int>, int> BuildCatToGroup(const std::vector<int> &mList) {
  std::unordered_map<int,int> cat2g;
  int K = static_cast<int>(mList.size()) - 1;
  for (int gi = 1; gi <= K; ++gi) {
    int L = mList[gi - 1];
    int R = mList[gi];
    int start = (L == 0) ? 1 : (L + 1);
    for (int c = start; c <= R; ++c) cat2g[c] = gi;
  }
  return {cat2g, K};
}

std::vector<Hist> MergePLogByGroup(const std::unordered_map<int, Hist> &plog,
                                   const std::unordered_map<int,int> &cat2g,
                                   int K) {
  std::vector<Hist> groups(K + 1);
  for (const auto &kv : plog) {
    int cat = kv.first;
    auto it = cat2g.find(cat);
    if (it == cat2g.end()) continue;
    int gi = it->second;
    if (gi < 1 || gi > K) continue;
    for (const auto &h : kv.second) {
      groups[gi][h.first] += h.second;
    }
  }
  return groups;
}

std::vector<double> GroupWeights(const std::vector<Hist> &groups) {
  int K = static_cast<int>(groups.size()) - 1;
  std::vector<double> totals(K + 1, 0.0);
  double sum = 0.0;
  for (int gi = 1; gi <= K; ++gi) {
    for (const auto &kv : groups[gi]) totals[gi] += kv.second;
    sum += totals[gi];
  }
  std::vector<double> w(K + 1, 0.0);
  if (sum <= 0) return w;
  for (int gi = 1; gi <= K; ++gi) w[gi] = totals[gi] / sum;
  return w;
}

std::vector<int> GroupBir(const std::vector<int> &mList,
                          const std::unordered_map<int,int> &birByCat) {
  int K = static_cast<int>(mList.size()) - 1;
  std::vector<int> BIR(K + 1, 0);
  for (int gi = 1; gi <= K; ++gi) {
    int L = mList[gi - 1];
    int R = mList[gi];
    int start = (L == 0) ? 1 : (L + 1);
    int best = 0;
    for (int c = start; c <= R; ++c) {
      auto it = birByCat.find(c);
      if (it != birByCat.end()) best = std::max(best, it->second);
    }
    BIR[gi] = best;
  }
  return BIR;
}

std::vector<double> RFromPLog(const std::vector<Hist> &groups, const std::vector<int> &BIR) {
  int K = static_cast<int>(groups.size()) - 1;
  std::vector<double> r(K + 1, 0.0);
  for (int gi = 1; gi <= K; ++gi) {
    const auto &hist = groups[gi];
    if (hist.empty() || BIR[gi] <= 0) { r[gi] = 0.0; continue; }
    double tot = 0.0, freeCnt = 0.0;
    for (const auto &kv : hist) {
      tot += kv.second;
      if (kv.first <= BIR[gi]) freeCnt += kv.second;
    }
    r[gi] = (tot <= 0) ? 0.0 : (freeCnt / tot);
  }
  return r;
}

std::vector<int> BuildRelocation(const std::unordered_map<int, std::vector<int>> &catSegs,
                                 const std::vector<int> &mList,
                                 const std::vector<int> &BIR,
                                 int maxCat) {
  std::vector<int> target(maxCat + 1, -1);
  int K = static_cast<int>(mList.size()) - 1;
  for (int c = 1; c <= maxCat; ++c) {
    int fromG = 1;
    for (int gi = 1; gi <= K; ++gi) {
      int L = mList[gi - 1];
      int R = mList[gi];
      int start = (L == 0) ? 1 : (L + 1);
      if (c >= start && c <= R) { fromG = gi; break; }
    }
    int birUpper = BIR[fromG];
    auto it = catSegs.find(c);
    int toG = std::min(fromG + 1, K);
    if (it != catSegs.end() && !it->second.empty()) {
      const auto &segList = it->second;
      std::vector<double> survivors;
      survivors.reserve(segList.size());
      for (int s : segList) {
        if (s >= birUpper) survivors.push_back(static_cast<double>(s));
      }
      if (!survivors.empty()) {
        std::vector<double> trimmed = survivors;
        std::sort(trimmed.begin(), trimmed.end());
        int n = static_cast<int>(trimmed.size());
        int kLow = 0;
        int kHigh = static_cast<int>(std::floor(n * 0.15));
        if (kLow + kHigh < n) {
          trimmed.erase(trimmed.begin(), trimmed.begin() + kLow);
          trimmed.erase(trimmed.end() - kHigh, trimmed.end());
        }
        if (!trimmed.empty()) {
          double geoSum = 0.0;
          double eps = 1e-9;
          for (double v : trimmed) geoSum += std::log(v + eps);
          double mean = std::exp(geoSum / trimmed.size()) - eps;
          double remaining = std::max(0.0, mean - birUpper);
          for (int gi = fromG; gi <= K; ++gi) {
            if (remaining <= static_cast<double>(BIR[gi])) { toG = gi; break; }
          }
        }
      }
    }
    target[c] = toG;
  }
  /*
  for (int i = 0; i < 10; i ++) {
	printf("TargetGC[%d]: G%d\n", i, target[i]);	
  } 
  */
  return target;
}

void ComputeInflows(int K,
                    const std::vector<double> &weights,
                    double pHot,
                    double hotInvalid,
                    const std::vector<double> &rCold,
                    double r1Eff,
                    std::vector<double> *coldDirect,
                    std::vector<double> *coldTotalIn,
                    std::vector<double> *totalIn) {
  coldDirect->assign(K + 1, 0.0);
  coldTotalIn->assign(K + 1, 0.0);
  totalIn->assign(K + 1, 0.0);

  double scaleCold = Clamp01(1.0 - pHot);
  for (int gi = 1; gi <= K; ++gi) {
    (*coldDirect)[gi] = scaleCold * weights[gi];
  }
  if (K >= 1) {
    (*coldTotalIn)[1] = (*coldDirect)[1];
    for (int gi = 2; gi <= K; ++gi) {
      (*coldTotalIn)[gi] = (*coldDirect)[gi] + (*coldTotalIn)[gi-1] * (1.0 - rCold[gi-1]);
    }
  }
  double inHotToG1 = pHot * (1.0 - hotInvalid);
  if (K >= 1) {
    (*totalIn)[1] = (*coldTotalIn)[1] + inHotToG1;
    for (int gi = 2; gi <= K; ++gi) (*totalIn)[gi] = (*coldTotalIn)[gi];
  }
}

double MeanExpDecay(int n, double k) {
  if (n <= 0) return 0.0;
  double s = 0.0;
  double invn = 1.0 / n;
  for (int j = 0; j < n; ++j) {
    double x = (j + 0.5) * invn;
    s += std::exp(-k * x);
  }
  return s / n;
}

double CalibrateK(int n, double targetOverStart) {
  double target = Clamp01(targetOverStart);
  if (n <= 0) return 0.0;
  if (std::fabs(target - 1.0) < 1e-6) return 0.0;
  if (target <= 1e-6) return 100.0;
  double lo = 0.0, hi = 100.0;
  for (int i = 0; i < 60; ++i) {
    double mid = 0.5 * (lo + hi);
    double m = MeanExpDecay(n, mid);
    if (m > target) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

double GroupValidBlocks(int nSegments, double startVal, double targetMean, int segBlocks=kSegBlocks) {
  if (nSegments <= 0) return 0.0;
  double tm = std::max(0.0, std::min(targetMean, startVal));
  double ratio = (startVal <= 0) ? 0.0 : (tm / startVal);
  double k = CalibrateK(nSegments, ratio);
  double s = 0.0;
  double invn = 1.0 / nSegments;
  for (int j = 0; j < nSegments; ++j) {
    double x = (j + 0.5) * invn;
    s += startVal * std::exp(-k * x);
  }
  return s * segBlocks;
}

std::vector<std::vector<double>> BuildMC(int K,
                                         const std::vector<double> &weights,
                                         double pHot,
                                         double hotInvalid,
                                         const std::vector<double> &rCold,
                                         double r1Eff,
                                         double lastVr) {
  int n = K + 2;
  std::vector<std::vector<double>> MC(n, std::vector<double>(n, 0.0));
  // Free row
  MC[0][1] = pHot;
  double remain = 1.0 - pHot;
  for (int gi = 1; gi <= K; ++gi) {
    MC[0][1 + gi] = remain * weights[gi];
  }
  // HOT row
  MC[1][0] = hotInvalid;
  if (K >= 1) MC[1][2] = 1.0 - hotInvalid;
  // G1..G(K-1)
  for (int gi = 1; gi < K; ++gi) {
    int row = 1 + gi;
    if (gi == 1) {
      MC[row][0] = r1Eff;
      MC[row][row + 1] = 1.0 - r1Eff;
    } else {
      MC[row][0] = rCold[gi];
      MC[row][row + 1] = 1.0 - rCold[gi];
    }
  }
  // GK
  if (K >= 1) {
    int rowk = 1 + K;
    MC[rowk][0] = 1.0 - lastVr;
    MC[rowk][rowk] = lastVr;
  }
  return MC;
}

double McPredictWaf(const std::vector<std::vector<double>> &MC, int totalSegmentsPhysical) {
  int n = static_cast<int>(MC.size());
  std::vector<double> base(n, 0.0);
  base[0] = static_cast<double>(totalSegmentsPhysical);
  double WAF = 1.0;
  for (int step = 0; step < 1000; ++step) {
    std::vector<double> nxt(n, 0.0);
    for (int i = 0; i < n; ++i) {
      double bi = base[i];
      if (bi == 0.0) continue;
      for (int j = 0; j < n; ++j) nxt[j] += bi * MC[i][j];
    }
    base.swap(nxt);
    if (step % 100 == 0) {
      double UW = base[0];
      double GW = 0.0;
      for (int idx = 2; idx < n; ++idx) {
        double UW_R = UW * MC[0][idx];
        GW += base[idx] - UW_R;
      }
      if (UW <= 1e-12) WAF = 1e9;
      else WAF = (UW + GW) / UW;
    }
  }
  return WAF;
}

EvalResult EvalConfig(const std::vector<int> &mList,
                      const std::unordered_map<int, Hist> &plog,
                      const std::unordered_map<int, std::vector<int>> &catSegs,
                      int hotSize,
                      double hotInvalid,
                      double pHot,
                      int birHot,
                      const std::unordered_map<int,int> &birByCat,
                      int logicalSizeGb,
                      double utilization) {
  EvalResult er;
  auto pair = BuildCatToGroup(mList);
  auto cat2g = pair.first;
  er.K = pair.second;
  er.mList = mList;
  auto groups = MergePLogByGroup(plog, cat2g, er.K);
  er.weights = GroupWeights(groups);
  er.BIR = GroupBir(mList, birByCat);
  // r from PLog
  auto rCold = RFromPLog(groups, er.BIR);
  // inflows
  double r1Eff = 0.0;
  std::vector<double> coldDirect, coldTotalIn, totalIn;
  if (er.K >= 1) {
    double inG1Total = 0.0, inG1Cold = 0.0;
    // temp compute r1Eff later after inflows
  }
  ComputeInflows(er.K, er.weights, pHot, hotInvalid, rCold, r1Eff, &coldDirect, &coldTotalIn, &totalIn);
  // recompute r1Eff with hot mix
  if (er.K >= 1) {
    double inHotToG1 = pHot * (1.0 - hotInvalid);
    double inG1Cold = coldTotalIn[1];
    double inG1Total = totalIn[1];
    if (inG1Total > 0) r1Eff = (inG1Cold * rCold[1] + inHotToG1 * 1.0) / inG1Total;
    else r1Eff = rCold[1];
  }
  // recompute inflows with updated r1Eff
  ComputeInflows(er.K, er.weights, pHot, hotInvalid, rCold, r1Eff, &coldDirect, &coldTotalIn, &totalIn);
  er.coldDirect = coldDirect;
  er.coldTotalIn = coldTotalIn;
  er.totalIn = totalIn;

  // capacity
  double logicalBlocks = logicalSizeGb * 1'000'000'000.0 / kPageBytes;
  int totalSegmentsPhysical = static_cast<int>(logicalSizeGb * 1'000'000'000.0 * kPhysLogicalFactor / kPageBytes / kSegBlocks);
  int usable = totalSegmentsPhysical - static_cast<int>(std::round(kOverheadPerGroupSeg * (er.K + 1)));
  if (usable < 2) usable = 2;

  // sizes
  std::vector<int> sizes(er.K + 1, 0);
  int used = hotSize;
  for (int gi = 1; gi < er.K; ++gi) {
    double raw = totalIn[gi] * er.BIR[gi];
    sizes[gi] = std::max(2, static_cast<int>(std::ceil(raw)));
    used += sizes[gi];
  }
  if (er.K >= 1) {
    sizes[er.K] = std::max(2, usable - used - 4);
  }

  // valid blocks
  double V_sum_prev = 0.0;
  if (hotSize > 0) {
    double targetMeanHot = std::max(0.0, 1.0 - hotInvalid);
    V_sum_prev += GroupValidBlocks(hotSize, 0.5, targetMeanHot, kSegBlocks);
  }
  for (int gi = 1; gi < er.K; ++gi) {
    int nseg = sizes[gi];
    double targetMean = (gi == 1) ? (1.0 - r1Eff) : (1.0 - rCold[gi]);
    V_sum_prev += GroupValidBlocks(nseg, 1.0, targetMean, kSegBlocks);
  }
  double V_total_logical = logicalBlocks * utilization;
  if (er.K >= 1) {
    double V_last = V_total_logical - V_sum_prev;
    if (sizes[er.K] <= 0) sizes[er.K] = 2;
    double denom_last = sizes[er.K] * kSegBlocks;
    if (V_last < 1e-6) V_last = 1e-6;
    double op = std::max(1e-6, std::min(0.999999, V_last / denom_last));
    double lastVr = FifoLastGroupVr(op);
    auto MC = BuildMC(er.K, er.weights, pHot, hotInvalid, rCold, r1Eff, lastVr);
    er.waf = McPredictWaf(MC, usable);
    er.ok = true;
  }
  return er;
}

bool IsValidTraffic(const EvalResult &er) {
  for (int gi = 1; gi <= er.K; ++gi) {
    if (er.coldDirect[gi] < kUserTrafficThreshold) return false;
  }
  return true;
}

} // namespace

bool OptimizeAndApplyGroupConfig(const std::string &modelName,
                                 double utilization,
                                 const std::string &baseDir) {
  std::string plogPath = baseDir + "/DOGI-PLog/" + modelName;
  std::string hotPath  = baseDir + "/DOGI-HOT/"  + modelName;
  std::string birPath  = baseDir + "/DOGI-BIR/"  + modelName;

  std::vector<PLogRecord> recs;
  int hotSize = 0;
  double hotInvalid = 0.0, pHot = 0.0;
  int birHot = 0;
  std::unordered_map<int,int> birByCat;

  if (!ReadPLog(plogPath, &recs)) return false;
  if (!ReadHot(hotPath, &hotSize, &hotInvalid, &pHot)) return false;
  if (!ReadBir(birPath, &birHot, &birByCat)) return false;

  std::unordered_map<int, Hist> plog;
  std::unordered_map<int, std::vector<int>> catSegs;
  int maxCat = 0;
  BuildPLog(recs, &plog, &catSegs, &maxCat);
  if (maxCat <= 0) return false;

  auto mLists = GenerateMLists(maxCat, kMaxGroups);
  if (mLists.empty()) return false;

  int logicalSizeGb = LogicalSizeGb; // global configured in main
  std::vector<EvalResult> results;
  results.reserve(mLists.size());
  for (const auto &m : mLists) {
    EvalResult er = EvalConfig(m, plog, catSegs, hotSize, hotInvalid, pHot, birHot, birByCat, logicalSizeGb, utilization);
    if (er.ok) results.push_back(er);
  }
  if (results.empty()) return false;

  // sort by WAF asc, tie by smaller K
  std::sort(results.begin(), results.end(), [](const EvalResult &a, const EvalResult &b) {
    if (a.waf == b.waf) return a.K < b.K;
    return a.waf < b.waf;
  });

  // Best-by-K summary (similar to python script)
  std::unordered_map<int, EvalResult> bestByK;
  for (const auto &er : results) {
    auto it = bestByK.find(er.K);
    if (it == bestByK.end() || er.waf < it->second.waf) {
      bestByK[er.K] = er;
    }
  }
  /*
  printf("==== Best configuration per K ====\n");
  for (int k = 1; k <= kMaxGroups; ++k) {
    auto it = bestByK.find(k);
    if (it == bestByK.end()) continue;
    const auto &br = it->second;
    printf("K=%d waf=%.4f m_list=", k, br.waf);
    for (size_t i = 0; i < br.mList.size(); ++i) {
      printf("%s%d", (i==0?"[":","), br.mList[i]);
    }
    printf("] BIR=[");
    for (int gi = 1; gi <= br.K; ++gi) {
      printf("%s%d", (gi==1?"":","), br.BIR[gi]);
    }
    printf("]\n");
  }
  */

  // filter traffic
  EvalResult chosen;
  bool found = false;
  for (const auto &er : results) {
    if (IsValidTraffic(er)) { chosen = er; found = true; break; }
  }
  if (!found) return false;

  auto relocation = BuildRelocation(catSegs, chosen.mList, chosen.BIR, maxCat);

  // Top-1 detail (like python Top-1)
  printf("==== Selected configuration ====\n");
  printf("WAF=%.4f K=%d m_list=", chosen.waf, chosen.K);
  for (size_t i = 0; i < chosen.mList.size(); ++i) {
    printf("%s%d", (i==0?"[":","), chosen.mList[i]);
  }
  printf("] BIR=[");
  for (int gi = 1; gi <= chosen.K; ++gi) {
    printf("%s%d", (gi==1?"":","), chosen.BIR[gi]);
  }
  printf("]\n");
  // GC relocation summary: category -> first GC target group
  if (!relocation.empty()) {
    printf("GC relocation (category -> first GC group): ");
    bool any = false;
    for (size_t c = 1; c < relocation.size(); ++c) {
      int tgt = relocation[c];
      if (tgt <= 0) continue;
      any = true;
      printf("c%zu->G%d ", c, tgt);
    }
    if (!any) printf("none");
    printf("\n");
  }

  uint64_t bir0 = BIR[0]; // preserve
  std::vector<uint64_t> birVals;
  birVals.push_back(bir0);
  // BIR indexed 1..K, we map to BIR[1..K-1] (skip last group)
  for (int gi = 1; gi < chosen.K; ++gi) {
    birVals.push_back(static_cast<uint64_t>(chosen.BIR[gi]));
  }
  SetGroupConfig(static_cast<uint32_t>(chosen.K), birVals, /*scaleUiToBlocks=*/true);
  {
    std::lock_guard<std::mutex> lk(g_mapMutex);
    g_mList = chosen.mList;
    g_firstGcTarget = relocation;
  }
  printf("[GCONF] applied from group optimizer: K=%d waf=%.4f m_list=", chosen.K, chosen.waf);
  for (size_t i = 0; i < chosen.mList.size(); ++i) {
    printf("%s%d", (i==0?"[":","), chosen.mList[i]);
  }
  printf("] BIR=[");
  for (int gi = 1; gi <= chosen.K; ++gi) {
    printf("%s%d", (gi==1?"":","), chosen.BIR[gi]);
  }
  printf("] (Unit: Segment)\n");
  //printf("naive_start: %d\n", naive_start);
  return true;
}

void SetCategoryMapping(const std::vector<int> &mList) {
  std::lock_guard<std::mutex> lk(g_mapMutex);
  g_mList = mList;
  g_firstGcTarget.clear();
}

int MapCategoryToGroup(int category) {
  //std::lock_guard<std::mutex> lk(g_mapMutex);
  if (!APPLY_ML) {
	if (category == 0) return 0;
	else return 1;
  } 
  //printf("APPLY_USER\n");
  if (category == 0) return 0;
  int K = static_cast<int>(g_mList.size()) - 1;
  for (int gi = 1; gi < K; ++gi) {
	int Gupper = g_mList[gi];
	if (category <= Gupper) return gi; 
  }
  /*
  //if (g_mList.size() < 2) return std::min(1, (int)NumGroup - 1);
  int K = static_cast<int>(g_mList.size()) - 1;
  for (int gi = 1; gi <= K; ++gi) {
    int L = g_mList[gi - 1];
    int R = g_mList[gi];
    int start = (L == 0) ? 1 : (L + 1);
    if (category >= start && category <= R) {
      return gi;
    }
  }
  */
  return K;
}

int MapCategoryFirstGc(int category, int currentGroup) {
  //std::lock_guard<std::mutex> lk(g_mapMutex);
  if (APPLY_ML == 0) {
	if (currentGroup >= (int)NumGroup - 2) return (int)NumGroup - 2; 
	else return currentGroup + 1; 
  }
  //printf("APPLY_ML\n");
  if (category == 0) return 1;
  if (currentGroup >= naive_start || category == -1) {
    if (currentGroup >= (int)NumGroup - 2) return (int)NumGroup - 2;
    else return currentGroup + 1;
  }
  else {
	return g_firstGcTarget[category]; 
  }

  if (category <= 0) return currentGroup;
  if (g_firstGcTarget.empty()) {
    int nextG = currentGroup + 1;
    if (nextG >= (int)NumGroup) nextG = (int)NumGroup - 1;
    return nextG;
  }
  if (category < static_cast<int>(g_firstGcTarget.size())) {
    int tgt = g_firstGcTarget[category];
    if (tgt > 0) return tgt;
  }
  if (g_mList.size() < 2) return currentGroup;
  int K = static_cast<int>(g_mList.size()) - 1;
  int nextG = currentGroup + 1;
  if (nextG > K) nextG = K;
  return nextG;
}
