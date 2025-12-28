// Minimal 2-layer MLP inference helper backed by OpenBLAS SGEMM.
#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <vector>
#include <string>

class MlpInference {
 public:
  static constexpr int kBatch = 128;
  static constexpr int kInputDim = 6;
  // Input feature order expected by the trained model:
  // [LBA, freq_bit, freq_bit2, interval_bit, seg_accessed, prev_lba]
  // The runtime collector provides [LBA, prev_lba, interval, freq_bit, freq_bit2, seg_accessed];
  // RunBatch permutes/normalizes internally, so callers can keep using the runtime order.
  static constexpr int kHidden1 = 32;
  static constexpr int kHidden2 = 32;
  static constexpr int kOutputDim = 10;

  MlpInference();
  ~MlpInference();

  // Load weights/biases + scaler (mean/scale) exported by model_trainer.py.
  // Files expected inside dir: W1.bin, W2.bin, W3.bin, b1.bin, b2.bin, b3.bin,
  // mean.bin, scale.bin (float32 row-major). Returns true on success.
  bool LoadFromDir(const std::string &dir, std::string *err = nullptr);
  bool IsModelLoaded() const { return weights_loaded_; }

  // Runs inference on up to kBatch items. If batch.size() < kBatch, the last
  // element is repeated to pad to a dense batch. Results are written into
  // argmaxOut (length kBatch).
  void RunBatch(const std::vector<std::array<float, kInputDim>> &batch,
                std::array<int, kBatch> &argmaxOut);
  void RunBatch(const std::vector<std::array<uint64_t, kInputDim>> &batch,
                std::array<int, kBatch> &argmaxOut);

 private:
  float *W1_ = nullptr;
  float *W2_ = nullptr;
  float *W3_ = nullptr;
  float *b1_ = nullptr;
  float *b2_ = nullptr;
  float *b3_ = nullptr;
  float *X_ = nullptr;
  float *Y1_ = nullptr;
  float *Y2_ = nullptr;
  float *Y3_ = nullptr;
  bool initialized_ = false;
  bool weights_loaded_ = false;
  bool scaler_loaded_ = false;
  std::mt19937 rng_;
  float mean_[kInputDim];
  float scale_[kInputDim];

  void AllocateBuffers();
  void InitWeights();
  void FreeBuffers();
  void InitScalerDefaults();
  bool LoadVector(const std::string &path, float *dst, size_t count, std::string *err);
  void NormalizeFeature(const float *raw, float *dst) const;
};
