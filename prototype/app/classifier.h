#ifndef APP_CLASSIFIER_H
#define APP_CLASSIFIER_H

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include "src/indexmap/filter.h"
#include "src/indexmap/factory.h"
#include "app/global.h"

// Tracks per-LBA estimated interval in units of segment_size*4.
class HotIntervalTracker {
public:
  static HotIntervalTracker& Instance();

  // Estimate interval at query time (saturating to 255).
  uint8_t Estimate(uint32_t blockAddr, uint64_t segmentAgeInBlocks) const;

  // Update rules
  void ResetOnUserWrite(uint32_t blockAddr);
  void AccumulateGcAge(uint32_t blockAddr, uint64_t segmentAgeInBlocks);

  static const uint64_t kSegmentBlocks; // from global config
  static const uint64_t kIntervalUnit;  // age unit

  static constexpr size_t kCapacity = 128ull * 1024ull * 1024ull; // matches Filter

private:
  HotIntervalTracker();
  uint64_t ToUnits(uint64_t ageInBlocks) const;
  uint8_t SaturatingAdd(uint8_t base, uint64_t deltaUnits) const;

  std::vector<uint8_t> intervals_;
};

// Tracks hotness for user writes using interval-based hot filter.
class DogiHotClassifier {
public:
  struct Result {
    uint8_t interval;
    bool isHot;
  };

  explicit DogiHotClassifier(uint64_t hotThreshold);
  // currentTimeInBlocks is a logical timestamp; age is computed as (current - creation_ts).
  bool IsHot(uint32_t blockAddr, uint64_t segmentAgeInBlocks);
  Result Classify(uint32_t blockAddr, uint64_t segmentAgeInBlocks);

private:
  HotIntervalTracker &tracker_;
  uint64_t hotThreshold_;
};

// Shared FrozenFilter used for GC classification (updated from main).
class FrozenFilterManager {
public:
  static FrozenFilterManager& Instance();

  void Update(uint32_t blockAddr);
  uint64_t Query(uint32_t blockAddr) const;

private:
  FrozenFilterManager();

  std::unique_ptr<IndexMap> filter_;
};

#endif // APP_CLASSIFIER_H
