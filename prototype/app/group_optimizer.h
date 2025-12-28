#ifndef APP_GROUP_OPTIMIZER_H
#define APP_GROUP_OPTIMIZER_H

#include <string>
#include <vector>

// Run PLog/HOT/BIR optimization (ported from 0914-DOGI-PLog.py) and apply
// group configuration (naive_start, BIR) in-place.
// - wk: workload name (e.g., "ycsb-a-mlp1")
// - utilization: logical utilization ratio (0..1)
// - baseDir: directory containing DOGI-PLog/{wk}, DOGI-HOT/{wk}, DOGI-BIR/{wk}
// Returns true on success, false otherwise.
// modelName: file name in DOGI-PLog / DOGI-HOT / DOGI-BIR (e.g., "Model7").
bool OptimizeAndApplyGroupConfig(const std::string &modelName,
                                 double utilization,
                                 const std::string &baseDir);

// Store category->group mapping derived from m_list, and query it.
// category 0 -> group 0 (hot). Categories start at 1.
void SetCategoryMapping(const std::vector<int> &mList);
int  MapCategoryToGroup(int category);

// First-GC relocation: category -> target group (based on PLog analysis).
// Returns -1 if unknown.
int  MapCategoryFirstGc(int category, int currentGroup);

#endif // APP_GROUP_OPTIMIZER_H
