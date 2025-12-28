#include <iostream>
#include <unistd.h>
#include <ctype.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <string>
#include <memory>
#include <array>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <utility>

#include "src/buse/buse.h"
#include "src/logstore/logstore.h"
#include "src/logstore/config.h"
#include "global.h"
#include "app/classifier.h"
#include "app/freq_features.h"
#include "app/group_config.h"
#include "app/group_optimizer.h"
#include "app/mlp_inference.h"
#include "app/model_train.h"

using namespace buse;

struct PendingWrite {
  off64_t start;
  size_t len;
  uint32_t lba;
  uint64_t interval;
  bool isHot;
};

struct BufferSlot {
  std::vector<PendingWrite> writes;
  size_t nonHotCount = 0;
  size_t bytes = 0;
  bool sealed = false;

  void reset() {
    writes.clear();
    nonHotCount = 0;
    bytes = 0;
    sealed = false;
  }
};

FILE *trace;
std::string latestModelName = ""; // default fallback for group config/PLog files
std::string latestModelDir;             // last successfully loaded model dir (empty if none)

// Resolve model directory and name (basename) using env or latest pointer.
static std::pair<std::string, std::string> ResolveModelDirAndName() {
  const char *envDir = std::getenv("MLP_MODEL_DIR");
  auto basename = [](const std::string &path) {
    auto pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
  };
  std::string modelDir;
  if (envDir) {
    modelDir = envDir;
  } else {
    const std::string latestPath = "../DOGI-Train/TrainedModel/latest";
    std::ifstream latest(latestPath);
    if (latest.good()) {
      std::string name;
      std::getline(latest, name);
      name.erase(name.find_last_not_of(" \t\r\n") + 1);
      if (!name.empty()) {
        modelDir = "../DOGI-Train/TrainedModel/" + name;
      }
    }
  }
  return {modelDir, basename(modelDir)};
}

int main(int argc, char *argv[]) {
//  Config::getInstance().selection = std::string(argv[1]);
  std::string resetCmd = std::string("zbd reset ") + kZnsDevicePath;
  system(resetCmd.c_str());
  printf("[SYSTEM] ZNS-SSD RESET SUCCESS\n");
  // Placement/selection are driven by the first CLI argument (e.g., DOGI); adjust here when wiring new policies.
  Config::GetInstance().placement = std::string(argv[1]);
  PlacementName = argv[1];
  if (strcmp(PlacementName, "DOGI") ==0) {
	Config::GetInstance().selection = "DogiSelect";
  }
  else Config::GetInstance().selection = "CostBenefit";
  printf("==============================\n");
  printf("             Setup            \n");
  printf("==============================\n");
  printf("Configuration Info\n");
  printf("PlacementName: %s\n", PlacementName);
  printf("Trace Path: %s\n", wk_name);
  printf("GpThreshold: %.4f, OpRatio: %.4f, LogicalSizeGb: %d\n", GpThreshold, OpRatio, LogicalSizeGb);
  printf("==============================\n");
  
  // hotThreshold still taken from argv if provided; default to 0 otherwise.
  uint64_t hotThreshold = 2;
  BIR[0] = hotThreshold * kSegmentBlocks;
  //printf("HotThreshold: %d\n", (int)hotThreshold);
  /*
  if(strcmp(PlacementName, "DOGI")==0){
    std::vector<uint64_t> birValues;
    for (int i = 3; i < argc; i++) {
      birValues.push_back(static_cast<uint64_t>(atol(argv[i])));
    }
    //SetGroupConfig(naive_start, birValues, true);
    for (size_t i = 0; i < birValues.size(); ++i) {
      printf("Group %zu BIR: %lu\n", i, BIR[i]);
    }
  }
  */
  int opt;
  const uint64_t deviceBytes =  (uint64_t)LogicalSizeGb * 1000 *1000 *1000ull * (1 + OpRatio);
  LogStore *logstore = new LogStore(deviceBytes, NumGroup);
  auto freqTracker = std::make_shared<FreqFeatures>(deviceBytes);
  ModelTrainer modelTrainer(deviceBytes);
  buseOperations *bop = logstore; // buse interface
  MlpInference mlpInference;
  // Try to load a pre-existing model (if any). If nothing found, stay uninitialized.
  {
    auto pair = ResolveModelDirAndName();
    std::string modelDir = pair.first;
    std::string modelName = pair.second;
    if (!modelDir.empty()) {
      std::string loadErr;
      if (mlpInference.LoadFromDir(modelDir, &loadErr)) {
        printf("[MLP] Loaded trained model from %s\n", modelDir.c_str());
        latestModelName = modelName;
        latestModelDir = modelDir;
      } else {
        printf("[MLP] Failed to load model from %s: %s\n", modelDir.c_str(), loadErr.c_str());
        printf("[MLP] No model loaded; using random weights until training completes.\n");
        latestModelDir.clear();
      }
    } else {
      printf("[MLP] No existing model found\n");
      latestModelDir.clear();
    }
  }


  /*
  alignas(512) char buffer[4096];
  for (int i = 0; i < 4096; ++i)
  {
    buffer[i] = 0;
  }
  */
  alignas(512) char buffer[4096 * 512];
  for (int i = 0; i < 4096 * 512; ++i)
  {
    buffer[i] = 0;
  }

  constexpr size_t kBufferBytes = 16 * 1024 * 1024;
  constexpr size_t kNonHotThreshold = 128;
  BufferSlot buffers[2];
  int activeIdx = 0;
  int inFlightIdx = -1;
  std::mutex bufMutex;
  std::condition_variable bufCv;
  std::atomic<bool> shutdown{false};
  uint64_t logicalTime = 0;
  uint64_t featureTime = 0;
  uint64_t totline = 0;
  const uint64_t passTimeBlocks = GetPassTimeBlocks();
  // Hot threshold is provided in units of segment_size*4 (same as tracker).
  DogiHotClassifier hotClassifier(hotThreshold);
  uint32_t prevLba = 0;

  auto sealActive = [&]() {
    std::unique_lock<std::mutex> lk(bufMutex);
    bufCv.wait(lk, [&]() { return inFlightIdx == -1; });
    buffers[activeIdx].sealed = true;
    inFlightIdx = activeIdx;
    bufCv.notify_one();
    activeIdx = 1 - activeIdx;
    buffers[activeIdx].reset();
  };

  // background worker: prepare ML batch for non-hot writes, then issue writes
  std::thread worker([&]() {
    uint64_t featureCount = 0;
    while (true) {
      BufferSlot local;
      {
        std::unique_lock<std::mutex> lk(bufMutex);
        bufCv.wait(lk, [&]() { return shutdown.load() || inFlightIdx != -1; });
        if (shutdown.load() && inFlightIdx == -1) break;
        int idx = inFlightIdx;
        inFlightIdx = -1;
        local = std::move(buffers[idx]);
        buffers[idx].reset();
        bufCv.notify_all();
      }
      // Build ML input batch (fixed 128 items, pad with last to keep it dense)
      static constexpr size_t kBatch = kNonHotThreshold;
      std::vector<std::array<uint64_t, 6>> mlBatch;
      mlBatch.reserve(kBatch);
      const size_t nonHotCount = local.nonHotCount;
      const size_t cappedNonHot = std::min(nonHotCount, static_cast<size_t>(MlpInference::kBatch));
      bool havePredictions = false;
      static thread_local std::array<int, MlpInference::kBatch> mlPred{};
      for (auto &w : local.writes) {
        uint8_t bits = freqTracker->GetFreqBits(w.lba);
        uint8_t bitCount = freqTracker->GetFreqBitCount(w.lba);
        uint64_t segFreq = freqTracker->GetChunkFrequency(w.lba);
        std::array<uint64_t, 6> feat = {
          static_cast<uint64_t>(w.lba),                  // current LBA
          static_cast<uint64_t>(prevLba),                // previous LBA
          static_cast<uint64_t>(w.interval),             // interval estimate (HotIntervalTracker units)
          static_cast<uint64_t>(bits),                   // freq_bit (recency-coded)
          static_cast<uint64_t>(bitCount),               // freq_bit2 (set bits)
          static_cast<uint64_t>(segFreq)                 // seg_accessed (2MiB chunk freq)
        };
        if (featureCount % 100000 == 0) {
          /*
          printf("[feature %llu] lba=%u prevLba=%u interval=%u freq_bits=%u freq_bitcnt=%u seg_freq=%lu\n",
                 (unsigned long long)featureCount,
                 w.lba,
                 prevLba,
                 (unsigned int)w.interval,
                 (unsigned int)bits,
                 (unsigned int)bitCount,
                 (unsigned long)segFreq);
          */
        }
        prevLba = w.lba;
        featureCount += 1;
		    featureTime += 1;
        if (featureTime > (deviceBytes*2 / 4096)) { // Trigger time of collecting training data
          modelTrainer.SaveTrainingData(&feat, featureTime, w.isHot);
        }
        if (w.isHot) continue; // skip hot block
        mlBatch.push_back(feat);
      }
      if (!mlBatch.empty() && mlpInference.IsModelLoaded()) {
        if (mlBatch.size() > MlpInference::kBatch) {
          mlBatch.resize(MlpInference::kBatch);
        }
        while (mlBatch.size() < kBatch) {
          mlBatch.push_back(mlBatch.back());
        }
        mlpInference.RunBatch(mlBatch, mlPred);
        havePredictions = true;
      }
      size_t predIdx = 0;
      for (auto &w : local.writes) {
        int category = w.isHot ? 0 : 1; // fallback when model not used
        if (!w.isHot && havePredictions && predIdx < cappedNonHot) {
          category = mlPred[predIdx]+1;
          predIdx++;
        }
        bop->write(buffer, w.len, w.start, category);
        freqTracker->OnBlockWrite(w.lba);
      }
    }
  });

  // background worker: run PLog/HOT/BIR optimizer once training is complete
  std::mutex configMutex;
  std::condition_variable configCv;
  std::atomic<bool> configDone{false};
  // Configuration: override GCONF_BASE_DIR to redirect optimized config output for artifacts.
  std::string gconfBase = std::getenv("GCONF_BASE_DIR") ? std::getenv("GCONF_BASE_DIR")
                                                        : "../DOGI-Gconf";
  std::thread configWorker([&]() {
    while (true) {
      std::unique_lock<std::mutex> lk(configMutex);
      configCv.wait(lk, [&]() { return shutdown.load() || modelTrainer.IsCollectingDone(); });
      //printf("Config worker woke up.\n");
      //printf("shutdown=%d, collectingDone=%d\n", shutdown.load(), modelTrainer.IsCollectingDone());
      //printf("!!!!!!!!!!!!!!!!!\n");
      if (shutdown.load()) break;
      lk.unlock();
      modelTrainer.MakingMLModel();
      // After training finishes, pick up the freshest model and reload.
      {
        auto pair = ResolveModelDirAndName();
        std::string modelDir = pair.first;
        std::string modelName = pair.second;
        if (!modelDir.empty()) {
          std::string loadErr;
          if (mlpInference.LoadFromDir(modelDir, &loadErr)) {
            latestModelName = modelName.empty() ? latestModelName : modelName;
            latestModelDir = modelDir;
            printf("[MLP] Reloaded trained model from %s\n", modelDir.c_str());
          } else {
            printf("[MLP] Reload failed for %s: %s\n", modelDir.c_str(), loadErr.c_str());
          }
        }
      }
	  double utilization = 1.0 - GpThreshold; // rough proxy
      bool ok = OptimizeAndApplyGroupConfig(latestModelName, utilization, gconfBase);
      //bool ok = false;
      configDone.store(ok);
      APPLY_ML = 1;
      //APPLY_ML = 0;
      printf("[GCONF] group optimizer %s\n", ok ? "success" : "failed");
    }
  });
  
  char line[100];

  off64_t start;
  uint32_t length;
  uint64_t end;

  {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    //printf("seconds: %ld micro seconds: %ld\n",
    //    current_time.tv_sec, current_time.tv_usec);
  }
  trace = fopen(wk_name, "r");
  while (fgets(line, sizeof(line), trace))
  {
	std::string str(line);
  std::vector<std::string> result;
    {
      std::stringstream ss(str);
      std::string substr;
	  while (ss >> substr) {
		result.push_back(substr);
      }
	}  
    if (result[1] != "1") continue;
    
    start = atoll(result[2].c_str())*4096;
    length = atoi(result[3].c_str());
    end = start + length;
    start = start & (~4095ull);
    length = ((end + 4095) & (~4095ull)) - start;
	uint32_t lba = (uint32_t)(start / 4096);
    logicalTime += 1;
    uint64_t segmentAge = logstore->GetSegmentAge(lba);
    if (logicalTime > passTimeBlocks) {
        FrozenFilterManager::Instance().Update(lba);
    }
    auto hotResult = hotClassifier.Classify(lba, segmentAge);
    bool isHot = (logicalTime > passTimeBlocks) ? hotResult.isHot : false;
    uint64_t interval = static_cast<uint64_t>(hotResult.interval);
    BufferSlot &bufSlot = buffers[activeIdx];
    bufSlot.writes.push_back({start, (size_t)length, lba, interval, isHot});
    if (!isHot) bufSlot.nonHotCount += 1;
    bufSlot.bytes += length;
    if (bufSlot.nonHotCount >= kNonHotThreshold || bufSlot.bytes >= kBufferBytes) {
      sealActive();
    }
    totline += 1;
    if (modelTrainer.IsCollectingDone()) {
      configCv.notify_one();
    }
	if (logicalTime > 786432000) break;
  }

  // flush remaining buffered work
  {
    std::unique_lock<std::mutex> lk(bufMutex);
    if (!buffers[activeIdx].writes.empty()) {
      buffers[activeIdx].sealed = true;
      inFlightIdx = activeIdx;
      bufCv.notify_one();
      activeIdx = 1 - activeIdx;
    }
    shutdown.store(true);
    bufCv.notify_all();
  }

  worker.join();
  shutdown.store(true);
  configCv.notify_one();
  configWorker.join();

  {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    printf("seconds: %ld micro seconds: %ld\n",
        current_time.tv_sec, current_time.tv_usec);

  }

  delete bop;
}
