#ifndef APP_FREQ_FEATURES_H
#define APP_FREQ_FEATURES_H

#include <cstdint>
#include <vector>
#include <mutex>

// Tracks per-LBA recency-weighted presence (8-bit), count of set bits,
// and per-2MiB chunk access counts. Intended for user writes only.
class FreqFeatures {
public:
  explicit FreqFeatures(uint64_t deviceBytes);

  // Update for a single 4KiB block.
  void OnBlockWrite(uint64_t blockAddr);

  // Recency-weighted score using the rotating 8-window bitmap
  // (current window weighted as 2^7, oldest as 2^0; max 255).
  uint8_t  GetFreqBits(uint64_t blockAddr);
  uint8_t  GetFreqBitCount(uint64_t blockAddr);
  uint64_t GetChunkFrequency(uint64_t blockAddr);

private:
  void advanceWindowIfNeeded(uint64_t blocksWritten);
  void advanceOneWindow();
  void clearCurrentWindowBit();
  void decayChunksIfNeeded(uint64_t blocksWritten);

  // Tunable windows for frequency features; adjust to align with device geometry/recency policy.
  static constexpr uint64_t kBlocksPerWindow = 8 * 1024ull * 1024ull;  // 32GiB of 4KiB blocks
  static constexpr uint64_t kBlocksPerChunk  = 512ull;                 // 2MiB / 4KiB
  static constexpr uint64_t kBlocksPerChunkDecay = 20ull * 1024ull * 1024ull * 1024ull / 4096ull; // 20GiB in blocks

  std::mutex mu_;
  uint64_t totalBlocks_;
  uint64_t totalChunks_;

  // Arrays sized by totalBlocks_/totalChunks_.
  std::vector<uint8_t> freqBits_;     // 8-bit recency pattern per LBA
  std::vector<uint64_t> chunkFreq_;   // per-2MiB chunk frequency

  uint64_t blocksSinceWindowStart_ = 0;
  uint8_t currentIdx_ = 0; // which bit represents the current window
  uint64_t blocksSinceChunkDecay_ = 0;
};

#endif // APP_FREQ_FEATURES_H
