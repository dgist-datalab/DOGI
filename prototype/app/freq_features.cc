#include "app/freq_features.h"

#include <algorithm>

FreqFeatures::FreqFeatures(uint64_t deviceBytes) {
  totalBlocks_ = deviceBytes / 4096;
  totalChunks_ = (totalBlocks_ + kBlocksPerChunk - 1) / kBlocksPerChunk;
  freqBits_.assign(totalBlocks_, 0);
  chunkFreq_.assign(totalChunks_, 0);
}

void FreqFeatures::advanceWindowIfNeeded(uint64_t blocksWritten) {
  blocksSinceWindowStart_ += blocksWritten;
  while (blocksSinceWindowStart_ >= kBlocksPerWindow) {
    blocksSinceWindowStart_ -= kBlocksPerWindow;
    advanceOneWindow();
  }
}

void FreqFeatures::advanceOneWindow() {
  currentIdx_ = static_cast<uint8_t>((currentIdx_ + 1) % 8);
  clearCurrentWindowBit();
}

void FreqFeatures::clearCurrentWindowBit() {
  uint8_t mask = static_cast<uint8_t>(~(1u << currentIdx_));
  for (auto &b : freqBits_) {
    b &= mask;
  }
}

void FreqFeatures::decayChunksIfNeeded(uint64_t blocksWritten) {
  blocksSinceChunkDecay_ += blocksWritten;
  if (blocksSinceChunkDecay_ < kBlocksPerChunkDecay) return;
  blocksSinceChunkDecay_ -= kBlocksPerChunkDecay;
  for (auto &c : chunkFreq_) {
    c = (c * 1) / 2; // decay to 2/3 of previous value
  }
}

void FreqFeatures::OnBlockWrite(uint64_t blockAddr) {
  //std::lock_guard<std::mutex> lk(mu_);
  advanceWindowIfNeeded(1);
  if (blockAddr >= totalBlocks_) return;
  freqBits_[blockAddr] |= static_cast<uint8_t>(1u << currentIdx_); // mark presence in current window
  uint64_t chunkIdx = blockAddr / kBlocksPerChunk;
  chunkFreq_[chunkIdx] += 1;
  decayChunksIfNeeded(1);
}

uint8_t FreqFeatures::GetFreqBits(uint64_t blockAddr) {
  //std::lock_guard<std::mutex> lk(mu_);
  if (blockAddr >= totalBlocks_) return 0;
  uint8_t bits = freqBits_[blockAddr];
  uint16_t score = 0; // avoid overflow during accumulation
  for (uint8_t i = 0; i < 8; ++i) {
    if ((bits & (1u << i)) == 0) continue;
    uint8_t distance = static_cast<uint8_t>((currentIdx_ + 8 - i) % 8); // 0 = current window
    uint8_t weight = static_cast<uint8_t>(1u << (7 - distance)); // exponential decay: 128..1
    score += weight; // max 255
  }
  if (score > 255) score = 255;
  return static_cast<uint8_t>(score);
}

uint8_t FreqFeatures::GetFreqBitCount(uint64_t blockAddr) {
  //std::lock_guard<std::mutex> lk(mu_);
  if (blockAddr >= totalBlocks_) return 0;
  uint8_t bits = freqBits_[blockAddr];
  return static_cast<uint8_t>(__builtin_popcount(static_cast<unsigned int>(bits)));
}

uint64_t FreqFeatures::GetChunkFrequency(uint64_t blockAddr) {
  //std::lock_guard<std::mutex> lk(mu_);
  if (blockAddr >= totalBlocks_) return 0;
  uint64_t chunkIdx = blockAddr / kBlocksPerChunk;
  return chunkFreq_[chunkIdx];
}
