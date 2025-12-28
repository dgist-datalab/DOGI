#include "app/classifier.h"

const uint64_t HotIntervalTracker::kSegmentBlocks = ::kSegmentBlocks;
const uint64_t HotIntervalTracker::kIntervalUnit  = HotIntervalTracker::kSegmentBlocks * 4;

HotIntervalTracker& HotIntervalTracker::Instance() {
  static HotIntervalTracker instance;
  return instance;
}

HotIntervalTracker::HotIntervalTracker() : intervals_(kCapacity, 0) {}

uint64_t HotIntervalTracker::ToUnits(uint64_t ageInBlocks) const {
  return ageInBlocks / kIntervalUnit;
}

uint8_t HotIntervalTracker::SaturatingAdd(uint8_t base, uint64_t deltaUnits) const {
  uint64_t sum = static_cast<uint64_t>(base) + deltaUnits;
  return static_cast<uint8_t>(std::min<uint64_t>(sum, 255ull));
}

uint8_t HotIntervalTracker::Estimate(uint32_t blockAddr, uint64_t segmentAgeInBlocks) const {
  if (blockAddr >= intervals_.size()) return 255;
  uint64_t delta = ToUnits(segmentAgeInBlocks);
  return SaturatingAdd(intervals_[blockAddr], delta);
}

void HotIntervalTracker::ResetOnUserWrite(uint32_t blockAddr) {
  if (blockAddr >= intervals_.size()) return;
  intervals_[blockAddr] = 0;
}

void HotIntervalTracker::AccumulateGcAge(uint32_t blockAddr, uint64_t segmentAgeInBlocks) {
  if (blockAddr >= intervals_.size()) return;
  uint64_t delta = ToUnits(segmentAgeInBlocks);
  intervals_[blockAddr] = SaturatingAdd(intervals_[blockAddr], delta);
}

DogiHotClassifier::DogiHotClassifier(uint64_t hotThreshold)
  : tracker_(HotIntervalTracker::Instance()), hotThreshold_(hotThreshold) {}

bool DogiHotClassifier::IsHot(uint32_t blockAddr, uint64_t segmentAgeInBlocks) {
  return Classify(blockAddr, segmentAgeInBlocks).isHot;
}

DogiHotClassifier::Result DogiHotClassifier::Classify(uint32_t blockAddr, uint64_t segmentAgeInBlocks) {
  uint8_t est = tracker_.Estimate(blockAddr, segmentAgeInBlocks);
  bool isHot = static_cast<uint64_t>(est) < hotThreshold_;
  
  tracker_.ResetOnUserWrite(blockAddr);
  return {est, isHot};
}

FrozenFilterManager& FrozenFilterManager::Instance() {
  static FrozenFilterManager instance;
  return instance;
}

FrozenFilterManager::FrozenFilterManager() {
  filter_ = std::unique_ptr<IndexMap>(IndexMapFactory::GetInstance("Filter"));
}

void FrozenFilterManager::Update(uint32_t blockAddr) {
  filter_->Update(blockAddr, (uint64_t)1);
}

uint64_t FrozenFilterManager::Query(uint32_t blockAddr) const {
  return filter_->Query(blockAddr);
}
