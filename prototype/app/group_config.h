#ifndef APP_GROUP_CONFIG_H
#define APP_GROUP_CONFIG_H

#include <cstdint>
#include <vector>
#include "app/global.h"
#include "app/classifier.h" // for HotIntervalTracker::kSegmentBlocks

// Update naive_start and BIR thresholds. NumGroup remains unchanged.
// If scaleUiToBlocks is true, each BIR value is scaled by segment_size (kSegmentBlocks blocks)
// to match the existing age units used in selection.
void SetGroupConfig(uint32_t newNaiveStart,
                    const std::vector<uint64_t> &birValues,
                    bool scaleUiToBlocks = true);

#endif // APP_GROUP_CONFIG_H
