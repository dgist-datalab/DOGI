#include "app/group_config.h"

#include <algorithm>

void SetGroupConfig(uint32_t newNaiveStart,
                    const std::vector<uint64_t> &birValues,
                    bool scaleUiToBlocks) {
  naive_start = newNaiveStart;

  constexpr size_t kMaxUi = sizeof(BIR) / sizeof(BIR[0]);
  size_t limit = std::min(birValues.size(), kMaxUi);
  for (size_t i = 0; i < limit; ++i) {
    uint64_t val = birValues[i];
    if (scaleUiToBlocks && i > 0) {
      val *= (HotIntervalTracker::kSegmentBlocks);
    }
    BIR[i] = val;
  }
  for (size_t i = limit; i < kMaxUi; ++i) {
    BIR[i] = 0;
  }
}
