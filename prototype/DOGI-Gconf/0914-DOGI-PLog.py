#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
WAF Predictor (PLog + HOT + BIR + Group Merge + FIFO last VR)
Usage:
    python waf_full.py {workload_name} {utilization}

Reads:
  DOGI-PLog/{workload}  -> CSV: idx,predicted_group,real_group,real_interval
  DOGI-HOT/{workload}   -> "HOT_size  HOT_invalid_ratio  HOT_traffic"
  DOGI-BIR/{workload}   -> one integer per line (first = HOT BIR), ends with -1

Notes:
  - predicted_group is 0-based in PLog; use category = predicted_group + 1 (1..N)
  - Segment size = 65536 blocks
  - Physical space = 1.1 * Logical space
"""

import sys, os, csv, math
from collections import Counter, defaultdict
from itertools import combinations
from collections import Counter
# ---------------------------- Tunables ----------------------------
MAX_GROUPS = 10          # 최대 그룹 수 (HOT 제외)
SEG_BLOCKS = 65536      # blocks per segment
PAGE_BYTES = 4096       # 4KB
PHYS_LOGICAL_FACTOR = 1.1  # Physical = 1.1 * Logical
BOUNDARY_TH = None      # e.g., 7; if None, no threshold on second-last boundary
GROUP_BIR_AGG = "max"   # group BIR from member categories: "max" or "mean"
USER_TRAFFIC_THRESHOLD = 0.0005
# 워크로드별 논리 용량(GB) 매핑 (없으면 128 사용)
WK_LOGICAL_SIZE_GB = {
    "ycsb-a-mlp1": 128, "ycsb-f-mlp1": 128, "var-mlp1": 128,
    "ali-126-mlp1": 50, "ali-293-mlp1": 120, "ali-132-mlp1": 50, "disk-2-mlp1": 40,
    # 필요 시 여기에 추가...
}

# 파일 상단 tunable에 추가해도 좋음
OVERHEAD_PER_GROUP_SEG = 0.5  # 그룹 1개당 못쓰는 세그먼트 수(기본 0.5)

def compute_usable_segments(total_segments_physical: int, K: int,
                            include_hot: bool = True,
                            overhead_per_group: float = OVERHEAD_PER_GROUP_SEG) -> int:
    """
    사용 가능 세그먼트 = 총 물리 세그먼트 - overhead_per_group * (#그룹수)
    #그룹수: HOT를 그룹으로 칠 거면 1+K, 아니면 K
    반환은 int로 바닥(Floor) 처리 (세그먼트는 정수)
    """
    group_count = (1 + K) if include_hot else K
    usable = total_segments_physical - overhead_per_group * group_count
    return max(2, int(usable))  # 최소 2 세그먼트 안전장치


# ------------------------- Math helpers --------------------------
def lambertw0(z: float) -> float:
    """Principal branch Lambert W (real part)."""
    try:
        from scipy.special import lambertw as _lw
        return float(_lw(z).real)
    except Exception:
        pass
    try:
        import mpmath as mp
        return float(mp.lambertw(z).real)
    except Exception:
        # crude Newton fallback for w*e^w = z
        w = 0.0
        for _ in range(80):
            ew = math.exp(w)
            f = w * ew - z
            df = ew * (w + 1.0)
            if abs(df) < 1e-16:
                break
            w_new = w - f / df
            if abs(w_new - w) < 1e-12:
                w = w_new
                break
            w = w_new
        return w

# ------------------------ File IO helpers ------------------------
def print_plog_category_counts(records):
    """
    records: [(idx, category, real_group, real_interval), ...]
    카테고리별 블록 개수 출력
    """
    counts = Counter(cat for _idx, cat, _rg, _ri in records)
    total = sum(counts.values())

    print("\n=== PLog Category Counts ===")
    print("Category  Count")
    print("-----------------")
    for cat in sorted(counts):
        print(f"c{cat:<7} {counts[cat]}")
    print("-----------------")
    print(f"TOTAL    {total}")

def _to_int(value):
    """Convert strings like '123' or '123.0' to int safely."""
    try:
        return int(value)
    except ValueError:
        return int(float(value))

def read_plog_records(path):
    """Return list of (idx:int, cat:int(1..), real_group:int, real_interval:int)."""
    out = []
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            idx = _to_int(row["idx"])
            cat = _to_int(row["predicted_group"]) + 1  # 0->1, ...
            rgrp = _to_int(row["real_group"])
            ri = _to_int(row["real_interval"])
            out.append((idx, cat, rgrp, ri))
    return out

def read_hot_info(path):
    """Return (hot_size:int segments, hot_invalid:float, p_hot:float)."""
    with open(path, "r") as f:
        txt = f.read().strip()
    parts = txt.replace(",", " ").split()
    if len(parts) < 3:
        raise ValueError(f"Invalid HOT file format at {path}")
    hot_size = int(float(parts[0]))
    hot_invalid = float(parts[1])
    p_hot = float(parts[2])
    # clamp
    hot_invalid = max(0.0, min(1.0, hot_invalid))
    p_hot = max(0.0, min(1.0, p_hot))
    return hot_size, hot_invalid, p_hot

def read_bir_list(path):
    """Return (bir_hot:int, bir_by_cat: dict{cat:int->bir:int})."""
    birs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                v = int(float(line))
            except:
                continue
            if v < 0:  # sentinel
                break
            birs.append(v)
    if not birs:
        return 0, {}
    bir_hot = birs[0]
    bir_by_cat = {}
    # The rest map to categories 1.. in order (if present).
    for i, v in enumerate(birs[1:], start=1):
        bir_by_cat[i] = int(v)
    return int(bir_hot), bir_by_cat

# ----------------------- PLog construction -----------------------
def build_category_plog(records, seg_blocks=SEG_BLOCKS):
    """Return dict{cat: Counter(seg_idx->count)} and max_cat."""
    plog = defaultdict(Counter)
    max_cat = 0
    for _, cat, _rg, ri in records:
        sidx = int(ri // seg_blocks)
        plog[cat][sidx] += 1
        if cat > max_cat: max_cat = cat
    return plog, max_cat

def build_category_segments(records, seg_blocks=SEG_BLOCKS):
    """
    카테고리별 segment index '리스트' (히스토그램 말고 원소 리스트)
    {cat: [sidx, sidx, ...]}
    """
    segs = defaultdict(list)
    max_cat = 0
    for _, cat, _rg, ri in records:
        sidx = int(ri // seg_blocks)
        segs[cat].append(sidx)
        if cat > max_cat: max_cat = cat
    return segs, max_cat

# ----------------------- m_list generation -----------------------
def generate_m_lists(max_cat, max_groups=MAX_GROUPS, boundary_th=BOUNDARY_TH):
    """
    Build candidates m_list: [0, b1, b2, ..., max_cat].
    Number of groups K = len(m_list)-1 (HOT 제외).
    1 <= K <= max_groups.
    Optional boundary_th: enforce b_{K-1} <= boundary_th (second last boundary).
    """
    inners = list(range(1, max_cat))
    out = []
    # K = number of groups
    for K in range(1, max_groups + 1):
        if K == 1:
            m = [0, max_cat]
            out.append(m)
            continue
        for comb in combinations(inners, K - 1):
            if sorted(comb) != list(comb):  # safety
                continue
            if boundary_th is not None and K >= 2:
                if comb[-1] > boundary_th:
                    continue
            m = [0] + list(comb) + [max_cat]
            out.append(m)
    return out

# ---------------------- Group-level utilities --------------------
def build_cat_to_group(m_list):
    """Map category->group_index(1..K)."""
    cat2g = {}
    K = len(m_list) - 1
    for gi in range(1, K + 1):
        L = m_list[gi - 1]
        R = m_list[gi]
        start = 1 if L == 0 else (L + 1)
        for c in range(start, R + 1):
            cat2g[c] = gi
    return cat2g, K

def merge_plog_by_groups(plog, cat2g, K):
    """Return list[Counter] groups[0..K], 0 dummy."""
    groups = [Counter() for _ in range(K + 1)]
    for cat, hist in plog.items():
        gi = cat2g.get(cat)
        if gi is not None and 1 <= gi <= K:
            groups[gi].update(hist)
    return groups

def group_weights_from_init(groups_init):
    """Cold weights for Free->Gi split, based on initial merged PLog."""
    K = len(groups_init) - 1
    totals = [0.0] * (K + 1)
    for gi in range(1, K + 1):
        totals[gi] = float(sum(groups_init[gi].values()))
    denom = sum(totals[1:])
    if denom <= 0:
        return [0.0] * (K + 1)
    return [0.0] + [totals[gi] / denom for gi in range(1, K + 1)]

def group_bir_from_categories(m_list, bir_by_cat, agg=GROUP_BIR_AGG):
    """Return list BIR[1..K]; 'agg' can be 'max' or 'mean'."""
    K = len(m_list) - 1
    BIR = [0] * (K + 1)
    for gi in range(1, K + 1):
        L = m_list[gi - 1]
        R = m_list[gi]
        start = 1 if L == 0 else (L + 1)
        vals = []
        for c in range(start, R + 1):
            v = bir_by_cat.get(c, 0)
            if v > 0: vals.append(v)
        if not vals:
            BIR[gi] = 0
        else:
            if agg == "mean":
                BIR[gi] = int(round(sum(vals) / len(vals)))
            else:
                BIR[gi] = max(vals)
    return BIR

def r_from_group_plog(groups_init, BIR):
    """
    For each group i, compute r_i_cold = (# s <= BIR_i)/total using initial merged PLog.
    Returns r_list[0..K] and totals[0..K].
    """
    K = len(groups_init) - 1
    r = [0.0] * (K + 1)
    totals = [0] * (K + 1)
    for gi in range(1, K + 1):
        hist = groups_init[gi]
        tot = sum(hist.values())
        totals[gi] = tot
        if tot == 0 or BIR[gi] <= 0:
            r[gi] = 0.0
            continue
        free_cnt = 0
        thr = BIR[gi]
        for s, cnt in hist.items():
            if s <= thr:
                free_cnt += cnt
        r[gi] = free_cnt / tot
    return r, totals

# ----------------- Inflow & size estimation (with HOT) -----------------
def compute_inflows(K, weights, p_hot, hot_invalid, r_cold):
    """
    Return (in_hot_to_g1, cold_direct[1..K], cold_total_in[1..K], total_in[1..K]).
    - cold_direct[i] = (1-p_hot)*weights[i]
    - cold_total_in[1] = cold_direct[1]
      cold_total_in[i] = cold_direct[i] + cold_total_in[i-1]*(1 - r_cold[i-1])  (i>=2)
    - in_hot_to_g1 = p_hot * (1 - hot_invalid)
    - total_in[1] = cold_total_in[1] + in_hot_to_g1
      total_in[i] = cold_total_in[i]  (i>=2; HOT은 G1에서 소멸)
    """
    cold_direct = [0.0] * (K + 1)
    cold_total_in = [0.0] * (K + 1)
    total_in = [0.0] * (K + 1)

    scale_cold = max(0.0, min(1.0, 1.0 - p_hot))
    for gi in range(1, K + 1):
        cold_direct[gi] = scale_cold * weights[gi]

    if K >= 1:
        cold_total_in[1] = cold_direct[1]
        for gi in range(2, K + 1):
            cold_total_in[gi] = cold_direct[gi] + cold_total_in[gi - 1] * (1.0 - r_cold[gi - 1])

    in_hot_to_g1 = p_hot * (1.0 - hot_invalid)

    if K >= 1:
        total_in[1] = cold_total_in[1] + in_hot_to_g1
        for gi in range(2, K + 1):
            total_in[gi] = cold_total_in[gi]

    return in_hot_to_g1, cold_direct, cold_total_in, total_in

def estimate_sizes(K, total_segments_physical, hot_size, BIR, total_in):
    """
    SIZE_HOT = hot_size (segments)
    SIZE_i   = ceil(total_in[i] * BIR[i])   for i=1..K-1
    SIZE_K   = remaining segments
    """
    sizes = [0] * (K + 1)
    used = hot_size
    for gi in range(1, max(1, K)):  # up to K-1
        raw = float(total_in[gi]) * float(BIR[gi])
        sizes[gi] = max(2, int(math.ceil(raw)))
        used += sizes[gi]
    # last group
    if K >= 1:
        sizes[K] = max(2, int(total_segments_physical - used - 4))
    return sizes

# ------------------ Valid profile & op calculation -------------------
def mean_exp_decay(n, k):
    """Mean of exp(-k*x_j) over j=0..n-1 with x_j=(j+0.5)/n."""
    if n <= 0:
        return 0.0
    s = 0.0
    invn = 1.0 / n
    for j in range(n):
        x = (j + 0.5) * invn
        s += math.exp(-k * x)
    return s / n

def calibrate_k_for_target_mean(n, target_over_start):
    """
    Find k >= 0 such that mean(exp(-k*x)) ~= target_over_start (in [0,1]).
    Binary search on k.
    """
    target = max(0.0, min(1.0, float(target_over_start)))
    if n <= 0:
        return 0.0
    # quick exits
    if abs(target - 1.0) < 1e-6:
        return 0.0  # k=0 -> mean=1
    if target <= 1e-6:
        return 100.0
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = mean_exp_decay(n, mid)
        if m > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def group_valid_blocks(n_segments, start_val, target_mean, seg_blocks=SEG_BLOCKS):
    """
    Build S(j)=start_val * exp(-k*x), choose k so that mean(S)=target_mean.
    Return total valid blocks: sum_j S(j) * seg_blocks.
    """
    if n_segments <= 0:
        return 0.0
    # target_mean must be <= start_val
    tm = max(0.0, min(float(target_mean), float(start_val)))
    ratio = 0.0 if start_val <= 0 else (tm / float(start_val))
    k = calibrate_k_for_target_mean(n_segments, ratio)
    # sum S
    s = 0.0
    invn = 1.0 / n_segments
    for j in range(n_segments):
        x = (j + 0.5) * invn
        s += start_val * math.exp(-k * x)
    return s * seg_blocks

def fifo_last_group_vr(op):
    """
    FIFO-based last group VR:
      x = 1/op, w = W0(-x e^{-x}), WAF = x/(x+w), VR = 1 - 1/WAF
    """
    op = max(1e-6, min(0.999999, float(op)))
    x = 1.0 / op
    z = -x * math.exp(-x)
    w = lambertw0(z)
    waf = x / (x + w) if (x + w) != 0 else 1e12
    vr = 1.0 - (1.0 / waf)
    # gentle clamps
    if vr >= 1.0:
        vr = 0.999
    if vr <= 0.05:
        vr = 0.8
    return vr

# -------------------- Markov & WAF estimation ----------------------
def build_MC(K, weights, p_hot, hot_invalid, r_cold, r1_eff, last_vr):
    """
    States: [Free(0), HOT(1), G1(2), ..., GK(K+1)]
    """
    n = K + 2
    MC = [[0.0 for _ in range(n)] for __ in range(n)]

    # Free row
    MC[0][1] = p_hot
    remain = 1.0 - p_hot
    for gi in range(1, K + 1):
        MC[0][1 + gi] = remain * weights[gi]

    # HOT row
    MC[1][0] = hot_invalid
    if K >= 1:
        MC[1][2] = 1.0 - hot_invalid  # to G1 only

    # G1..G(K-1)
    for gi in range(1, K):
        row = 1 + gi
        if gi == 1:
            MC[row][0] = r1_eff
            MC[row][row + 1] = 1.0 - r1_eff
        else:
            MC[row][0] = r_cold[gi]
            MC[row][row + 1] = 1.0 - r_cold[gi]

    # GK
    if K >= 1:
        rowk = 1 + K
        MC[rowk][0] = 1.0 - last_vr
        MC[rowk][rowk] = last_vr

    return MC

def mc_predict_waf(MC, total_segments_physical, steps=1000, log_term=100):
    """
    Roughly follow original MC_STEP idea.
    """
    n = len(MC)
    base = [0.0] * n
    base[0] = float(total_segments_physical)  # all free initially

    WAF = 1.0
    for step in range(steps):
        nxt = [0.0] * n
        for i in range(n):
            bi = base[i]
            if bi == 0.0: continue
            row = MC[i]
            for j in range(n):
                nxt[j] += bi * row[j]
        base = nxt

        if step % log_term == 0:
            UW = base[0]
            GW = 0.0
            for idx in range(2, n):
                UW_R = UW * MC[0][idx]  # portion that came from Free this tick
                GW += base[idx] - UW_R
            if UW <= 1e-12:
                WAF = float("inf")
            else:
                WAF = (UW + GW) / UW
    return WAF

# ------------------------ Configuration eval ----------------------
def eval_configuration(wk, utilization, plog, max_cat, hot_size, hot_invalid, p_hot,
                       bir_hot, bir_by_cat, m_list,
                       logical_size_gb):
    cat2g, K = build_cat_to_group(m_list)
    groups_init = merge_plog_by_groups(plog, cat2g, K)
    weights = group_weights_from_init(groups_init)
    BIR = group_bir_from_categories(m_list, bir_by_cat, agg=GROUP_BIR_AGG)

    # r_i from PLog (cold)
    r_cold, _totals = r_from_group_plog(groups_init, BIR)

    # inflows (rates)
    in_hot_to_g1, cold_direct, cold_total_in, total_in = compute_inflows(
        K, weights, p_hot, hot_invalid, r_cold
    )

    # r1_eff: mix cold G1 + hot->G1(=100% invalid in G1)
    if K >= 1:
        in_g1_total = total_in[1]
        in_g1_cold = cold_total_in[1]
        if in_g1_total > 0:
            r1_eff = (in_g1_cold * r_cold[1] + in_hot_to_g1 * 1.0) / in_g1_total
        else:
            r1_eff = r_cold[1]
    else:
        r1_eff = 0.0

    # capacity (segments)
    logical_blocks = logical_size_gb * 1_000_000_000.0 / PAGE_BYTES
    total_segments_physical = int(logical_size_gb * 1_000_000_000.0 * PHYS_LOGICAL_FACTOR / PAGE_BYTES / SEG_BLOCKS)

    total_segments_usable = compute_usable_segments(total_segments_physical, K, include_hot=True)

    # sizes
    sizes = estimate_sizes(K, total_segments_usable, hot_size, BIR, total_in)
    #sizes = estimate_sizes(K, total_segments_physical, hot_size, BIR, total_in)

    # valid blocks for HOT..G_{K-1} via requested shape (fast early drop)
    V_sum_prev = 0.0
    if hot_size > 0:
        target_mean_hot = max(0.0, 1.0 - hot_invalid)  # proportion going to G1
        V_hot = group_valid_blocks(hot_size, start_val=0.5, target_mean=target_mean_hot, seg_blocks=SEG_BLOCKS)
        V_sum_prev += V_hot

    for gi in range(1, max(1, K)):  # up to K-1
        nseg = sizes[gi]
        if nseg <= 0:
            continue
        if gi == 1:
            target_mean = max(0.0, 1.0 - r1_eff)
        else:
            target_mean = max(0.0, 1.0 - r_cold[gi])
        V_i = group_valid_blocks(nseg, start_val=1.0, target_mean=target_mean, seg_blocks=SEG_BLOCKS)
        V_sum_prev += V_i

    # last group valid blocks from logical target
    V_total_logical = logical_blocks * float(utilization)
    V_last = V_total_logical - V_sum_prev
    # guard
    if sizes[K] <= 0:
        sizes[K] = 2
    denom_last = sizes[K] * SEG_BLOCKS
    if V_last < 1e-6:
        V_last = 1e-6
    op = V_last / float(denom_last)
    op = max(1e-6, min(0.999999, op))

    last_vr = fifo_last_group_vr(op)

    # MC & WAF
    MC = build_MC(K, weights, p_hot, hot_invalid, r_cold, r1_eff, last_vr)
    waf = mc_predict_waf(MC, total_segments_usable, steps=1000, log_term=100)

    return {
        "total_in": total_in,                 # 🔹 각 그룹 유입 비율 (Free 기준 전체 비율)
        "cold_total_in": cold_total_in,       # (참고) cold만의 재귀 유입
        "cold_direct": cold_direct,           # (참고) Free→Gi (cold) 직접 유입
        "bir_hot": bir_hot,                   # 🔹 HOT BIR (참고)
        "wk": wk,
        "waf": waf,
        "K": K,
        "m_list": m_list,
        "BIR": BIR,
        "hot_size": hot_size,
        "hot_invalid": hot_invalid,
        "p_hot": p_hot,
        "weights": weights,
        "r_cold": r_cold,
        "r1_eff": r1_eff,
        "sizes": sizes,
        "op_last": op,
        "last_vr": last_vr,
        "total_segments_physical": total_segments_physical,
        "logical_blocks": logical_blocks,
    
        # 🔹 테이블/출력용으로 누락되면 KeyError 나는 필드들:
        "MC": MC,                         # ← 반드시 추가
        "total_in": total_in,             # Free 기준 Gi 유입 비율
        "cold_total_in": cold_total_in,   # (참고) cold 재귀 유입
        "cold_direct": cold_direct,       # (참고) Free→Gi(cold) 직접 유입
        "bir_hot": bir_hot,               # (참고) HOT BIR
    }

def print_group_transition_table(res):
    """
    Print table:
    Group | BIRupper | ValidRatio | UserTraffic | HOT | G1 | G2 | ... | GK
    - ValidRatio: HOT=1-hot_invalid, G1=1-r1_eff, G2..G(K-1)=1-r_cold[i], GK=last_vr
    - UserTraffic: Free→state 비율 (HOT=p_hot, Gi=cold_direct[i] = (1-p_hot)*weights[i])
    - Transition columns: 각 행(그룹/HOT)에서 HOT..GK로 가는 확률(MC의 해당 row에서 1..K+1열)
      (지시대로 Free 컬럼은 제외)
    """
    K = res["K"]
    MC = res["MC"]
    BIR = res["BIR"]
    p_hot = res["p_hot"]
    hot_invalid = res["hot_invalid"]
    last_vr = res["last_vr"]
    r_cold = res["r_cold"]
    r1_eff = res["r1_eff"]
    bir_hot = res.get("bir_hot", 0)
    cold_direct = res["cold_direct"]   # ★ UserTraffic source

    # 헤더
    dest_cols = ["HOT"] + [f"G{i}" for i in range(1, K+1)]
    header = ["Group", "BIRupper", "ValidRatio", "UserTraffic"] + dest_cols
    col_widths = [max(len(h), 10) for h in header]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    print("\nGroup BIRupper ValidRatio UserTraffic and transitions (excluding Free):")
    print(fmt_row(header))
    print(fmt_row(["-"*len(h) for h in header]))

    # 1) HOT 행 (state index = 1)
    hot_valid = max(0.0, 1.0 - hot_invalid)
    hot_row = ["HOT", bir_hot, f"{hot_valid:.4f}", f"{p_hot:.6f}"]
    # MC row for HOT is index 1; columns HOT..GK are indices 1..K+1
    hot_trans = [f"{MC[1][1 + j]:.4f}" for j in range(0, K+1)]  # 0:HOT, 1:G1, ..., K:GK
    hot_row += hot_trans
    print(fmt_row(hot_row))

    # 2) G1..G(K-1)
    for gi in range(1, K):
        label = f"G{gi}"
        valid = (1.0 - r1_eff) if gi == 1 else (1.0 - r_cold[gi])
        bir_up = BIR[gi]
        user_traffic = cold_direct[gi]           # ★ Free→Gi
        row = [label, bir_up, f"{valid:.4f}", f"{user_traffic:.6f}"]
        row_trans = [f"{MC[1 + gi][1 + j]:.4f}" for j in range(0, K+1)]
        row += row_trans
        print(fmt_row(row))

    # 3) GK
    if K >= 1:
        label = f"G{K}"
        valid = last_vr
        bir_up = BIR[K]
        user_traffic = cold_direct[K]            # ★ Free→GK
        row = [label, bir_up, f"{valid:.4f}", f"{user_traffic:.6f}"]
        row_trans = [f"{MC[1 + K][1 + j]:.4f}" for j in range(0, K+1)]
        row += row_trans
        print(fmt_row(row))
    print()


from typing import Dict, List, Any, Optional
import math


from typing import Dict, List, Any, Optional
import math

import math
from typing import Dict, List, Any, Optional

def compute_category_relocation_by_mean(
    cat_segments: Dict[int, List[int]],
    m_list: List[int],
    BIR: List[float],
    trim_low: float = 0.05,
    trim_high: float = 0.05,
    active_mask: Optional[List[bool]] = None,
    to_blocks: bool = False,
    blocks_per_segment: int = 65536,
    mean_mode: str = "arith",         # ★ "arith" 또는 "geo"
    geo_eps: float = 1e-9,            # ★ 기하평균용 epsilon
) -> List[Dict[str, Any]]:

    def _build_cat_to_group(m_list: List[int]):
        cat2g = {}
        K = len(m_list) - 1
        for gi in range(1, K + 1):
            L = m_list[gi - 1]; R = m_list[gi]
            start = 1 if L == 0 else (L + 1)
            for c in range(start, R + 1):
                cat2g[c] = gi
        return cat2g, K

    def _trim_slice(vs: List[float], low_frac: float, high_frac: float) -> List[float]:
        n = len(vs)
        if n == 0: return []
        vs = sorted(vs)
        k_low = int(max(0, min(n-1, math.floor(n * max(0.0, min(low_frac, 0.49))))))
        k_high = int(max(0, min(n-1, math.floor(n * max(0.0, min(high_frac, 0.49))))))
        if k_low + k_high >= n:
            return vs  # 트리밍 과도 → 원본 유지
        return vs[k_low:n-k_high]

    def _mean_value(vals: List[float]) -> Optional[float]:
        if not vals: return None
        if mean_mode == "geo":
            # 기하평균: log(x+eps)의 평균을 지수화
            # 0 처리 위해 eps 시프트
            s = 0.0
            for x in vals:
                y = x + geo_eps
                if y <= 0:  # 예외 안전장치
                    return None
                s += math.log(y)
            return math.exp(s / len(vals)) - geo_eps
        else:
            # 기본: 산술평균
            return sum(vals) / len(vals)

    def _first_active_from(start_g: int, K: int, active_mask: Optional[List[bool]]) -> int:
        if active_mask is None:
            return min(start_g, K)
        for h in range(start_g, K + 1):
            if active_mask[h]: return h
        for h in range(K, 0, -1):
            if active_mask[h]: return h
        return K

    def _pick_target_group(g_start: int, remaining: float, K: int, BIR: List[float],
                           active_mask: Optional[List[bool]]) -> int:
        for h in range(g_start, K + 1):
            if active_mask is not None and not active_mask[h]:
                continue
            if remaining <= float(BIR[h]):
                return h
        if active_mask is not None:
            for h in range(K, g_start - 1, -1):
                if active_mask[h]: return h
        return K

    cat2g, K = _build_cat_to_group(m_list)
    max_cat = m_list[-1]
    results: List[Dict[str, Any]] = []

    for c in range(1, max_cat + 1):
        s_list = cat_segments.get(c, [])
        g_start = cat2g.get(c, K if K >= 1 else 1)
        BIR_g = float(BIR[g_start]) if g_start <= K else 0.0

        if not s_list:
            # 데이터 없는 카테고리: 무조건 다음 그룹(마지막이면 K)
            desired = g_start + 1 if g_start < K else K
            to_group = _first_active_from(desired, K, active_mask)
            mean_life = None
            remaining = None
            surv_total = 0
            used = 0
        else:
            # 생존 표본(≥ BIR_g)
            survivors = [float(s) for s in s_list if float(s) >= BIR_g]
            surv_total = len(survivors)
            if surv_total == 0:
                mean_life = None
                remaining = None
                to_group = g_start
                used = 0
            else:
                trimmed = _trim_slice(survivors, trim_low, trim_high)
                if not trimmed:
                    trimmed = survivors
                mean_life = _mean_value(trimmed)  # 산술 or 기하
                if mean_life is None:
                    remaining = None
                    to_group = g_start
                    used = 0
                else:
                    remaining = max(0.0, mean_life - BIR_g)
                    if remaining != None:
                        remaining = remaining*0.95
                    used = len(trimmed)
                    to_group = _pick_target_group(g_start, remaining, K, BIR, active_mask)

        # 단위 변환
        bir_upper = BIR_g
        ml_out = mean_life
        rm_out = remaining
        unit = "segments"
        if to_blocks:
            bir_upper *= blocks_per_segment
            ml_out = None if mean_life is None else mean_life * blocks_per_segment
            rm_out = None if remaining is None else remaining * blocks_per_segment
            unit = "blocks"

        results.append({
            "cat": c,
            "from_group": g_start,
            "to_group": to_group,
            "bir_upper": bir_upper,
            "mean_life": ml_out,
            "remaining_life": rm_out,
            "survivors_used": used,
            "survivors_total": surv_total,
            "unit": unit,
            "mean_mode": mean_mode,
        })

    return results



def print_category_relocation_table(stats: List[Dict[str, Any]]):
    """compute_category_relocation_by_mean() 결과를 사람이 보기 좋게 출력."""
    if not stats:
        print("\n(no category relocation stats)\n")
        return
    unit = stats[0].get("unit", "segments")
    header = ["cat", "fromG", "toG", f"BIR_upper({unit})", f"mean({unit})", f"remain({unit})", "n_used/n_surv"]
    widths = [max(len(h), 8) for h in header]

    def fmt_row(cells):
        return "  ".join(str(x).ljust(w) for x, w in zip(cells, widths))

    print(f"\n=== Category Relocation by Mean (trimmed) [{unit}] ===")
    print(fmt_row(header))
    print(fmt_row(["-" * len(h) for h in header]))
    for d in stats:
        cat = f"c{d['cat']}"
        fg = f"G{d['from_group']}"
        tg = f"G{d['to_group']}"
        bu = "-" if d["bir_upper"] is None else f"{d['bir_upper']:.3f}"
        ml = "-" if d["mean_life"] is None else f"{d['mean_life']:.3f}"
        rm = "-" if d["remaining_life"] is None else f"{d['remaining_life']:.3f}"
        n_used = d["survivors_used"]; n_all = d["survivors_total"]
        print(fmt_row([cat, fg, tg, bu, ml, rm, f"{n_used}/{n_all}"]))
    print()


def scale_bir_for_display(v: int) -> float:
    """BIR 출력용 스케일링."""
    if v <= 5:
        return v 
    elif v <= 15:
        return v*0.85
    else:
        return v * 0.7
    """
    if v <= 55:
        return v * 0.9 
    elif v <= 110:
        return v * 0.8
    elif v <= 160:
        return v *0.75
    else:
        return v * 0.7
    """
def is_config_valid_no_zero_traffic(res) -> bool:
    """
    total_in[1..K] 중 하나라도 EPS 이하(≈0)이면 그 구성은 스킵.
    """
    K = res["K"]
    tin = res["total_in"]
    for gi in range(1, K+1):
        if tin[gi] <= EPS_TRAFFIC:
            return False
    return True

def is_config_valid_user_traffic(res, thr: float) -> bool:
    """
    G1..GK 중 user-traffic(=cold_direct[gi])이 thr 미만인 그룹이 하나라도 있으면 False.
    HOT은 제외.
    """
    K = res["K"]
    cd = res["cold_direct"]
    for gi in range(1, K + 1):
        if cd[gi] < thr:
            return False
    return True

# ----------------------------- Main -------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python waf_full.py {workload_name} {utilization}")
        sys.exit(1)

    wk = sys.argv[1].strip()
    utilization = float(sys.argv[2])

    # paths
    plog_path = os.path.join("DOGI-PLog", wk)
    hot_path  = os.path.join("DOGI-HOT", wk)
    bir_path  = os.path.join("DOGI-BIR", wk)

    if not os.path.exists(plog_path):
        raise FileNotFoundError(f"Missing PLog file: {plog_path}")
    if not os.path.exists(hot_path):
        raise FileNotFoundError(f"Missing HOT file: {hot_path}")
    if not os.path.exists(bir_path):
        raise FileNotFoundError(f"Missing BIR file: {bir_path}")

    # read inputs
    print("READ INPUT")
    print("READ Output for building PLog")
    records = read_plog_records(plog_path)
    print("READ HOT INFO")
    hot_size, hot_invalid, p_hot = read_hot_info(hot_path)
    print("READ BIR INFO")
    bir_hot, bir_by_cat = read_bir_list(bir_path)

    # PLog
    plog, max_cat_from_data = build_category_plog(records, seg_blocks=SEG_BLOCKS)
    print_plog_category_counts(records)
    cat_segments, _ = build_category_segments(records, seg_blocks=SEG_BLOCKS)  # 🔹 추가 
    max_cat = max(max_cat_from_data, max(bir_by_cat.keys() or [0]))

    # workload logical size (GB)
    if "disk" in wk:
        logical_size_gb = WK_LOGICAL_SIZE_GB.get(wk, 40)
    elif "ali" in wk:
        logical_size_gb = WK_LOGICAL_SIZE_GB.get(wk, 137.433)
    else: 
        logical_size_gb = WK_LOGICAL_SIZE_GB.get(wk, 128)
    # m_list candidates
    mlists = generate_m_lists(max_cat, max_groups=MAX_GROUPS, boundary_th=BOUNDARY_TH)
    if not mlists:
        raise RuntimeError("No m_list candidates generated. Check max_cat or constraints.")

    # evaluate each
    results = []
    for m in mlists:
        try:
            res = eval_configuration(
                wk=wk,
                utilization=utilization,
                plog=plog,
                max_cat=max_cat,
                hot_size=hot_size,
                hot_invalid=hot_invalid,
                p_hot=p_hot,
                bir_hot=bir_hot,
                bir_by_cat=bir_by_cat,
                m_list=m,
                logical_size_gb=logical_size_gb,
            )
            results.append(res)
        except Exception as e:
            # skip invalid cand
            continue

    if not results:
        raise RuntimeError("No valid configurations evaluated.")


    # sort by WAF asc (tie-breaker: fewer groups first)
    results.sort(key=lambda r: (r["waf"], r["K"]))

    # 🔹 user-traffic 기준으로 구성 필터링
    filtered = [r for r in results if is_config_valid_user_traffic(r, USER_TRAFFIC_THRESHOLD)]
    skipped = len(results) - len(filtered)
    if skipped > 0:
        print(f"(filtered out {skipped} configurations due to low user-traffic groups < {USER_TRAFFIC_THRESHOLD:.6f})\n")

    if not filtered:
        print("No configurations left after user-traffic filtering.")
        return

    # =========================
    # 🔹 NEW: Best-by-K summary
    # =========================
    best_by_K = {}
    for r in filtered:
        K = r["K"]
        if (K not in best_by_K) or (r["waf"] < best_by_K[K]["waf"]):
            best_by_K[K] = r

    print("=" * 80)
    print("Best configuration for each group count (after user-traffic filtering)")
    print("=" * 80)
    for K in sorted(best_by_K.keys()):
        br = best_by_K[K]
        scaled_bir = [round(scale_bir_for_display(v), 2) for v in br["BIR"][1:]]
        print(f"K={K} -> WAF={br['waf']:.4f} | m_list={br['m_list']} | BIR={br['BIR'][1:]} (scaled)")
        #print(f"K={K} -> WAF={br['waf']:.4f} | m_list={br['m_list']} | BIR={scaled_bir} (scaled)")

    # 기존 Top-4도 그대로 유지하려면 아래 계속 실행
    topk = filtered[:4]
    print("=" * 80)
    print(f"Top-{len(topk)} configurations by predicted WAF (lowest first)")
    print("=" * 80)
    for rank, r in enumerate(topk, 1):
        print(f"[{rank}] WAF={r['waf']:.4f} | K={r['K']} | m_list={r['m_list']}")
        print(f"     BIR (per group 1..K): {r['BIR'][1:]}")
        scaled_bir = [round(scale_bir_for_display(v), 2) for v in r["BIR"][1:]]
        print(f"     BIR (per group 1..K): {scaled_bir} (scaled for display)")
        print_group_transition_table(r)

        print("Per-category GC decision:")
        print(f"  HOT -> G1 (fixed)  | invalid={r['hot_invalid']*100:.1f}%")

        relos = compute_category_relocation_by_mean(
            cat_segments=cat_segments,
            m_list=r["m_list"],
            BIR=r["BIR"],
            trim_low=0, trim_high=0.15,
            active_mask=r.get("active_mask"),
            to_blocks=False,
            blocks_per_segment=65536,
            mean_mode="geo",
            geo_eps=1e-9,
        )
        print_category_relocation_table(relos)
        print("-" * 80)


if __name__ == "__main__":
    main()
