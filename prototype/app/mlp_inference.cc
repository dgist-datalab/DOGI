#include "app/mlp_inference.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>

#include <cblas.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {

// The training pipeline expects features in this order:
// [LBA, freq_bit, freq_bit2, interval_bit, seg_accessed, prev_lba]
// The runtime feature collector builds: [LBA, prev_lba, interval, freq_bit, freq_bit2, seg_accessed]
// Use this permutation to map runtime->model order.
constexpr int kFeaturePermute[MlpInference::kInputDim] = {0, 3, 4, 2, 5, 1};

void *aligned_malloc64(size_t bytes) {
  void *p = nullptr;
  if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
  return p;
}

void aligned_free(void *p) { free(p); }

inline void bias_relu_inplace(float *Y, const float *bias, int rows, int cols) {
#if defined(__AVX512F__)
  const __m512 vzero = _mm512_setzero_ps();
  for (int i = 0; i < rows; i++) {
    float *row = Y + static_cast<size_t>(i) * cols;
    int j = 0;
    for (; j + 15 < cols; j += 16) {
      __m512 vy = _mm512_loadu_ps(row + j);
      __m512 vb = _mm512_loadu_ps(bias + j);
      vy = _mm512_add_ps(vy, vb);
      vy = _mm512_max_ps(vy, vzero);
      _mm512_storeu_ps(row + j, vy);
    }
    for (; j < cols; j++) {
      float v = row[j] + bias[j];
      row[j] = (v > 0.f) ? v : 0.f;
    }
  }
#else
  for (int i = 0; i < rows; i++) {
    float *row = Y + static_cast<size_t>(i) * cols;
    for (int j = 0; j < cols; j++) {
      float v = row[j] + bias[j];
      row[j] = (v > 0.f) ? v : 0.f;
    }
  }
#endif
}

inline void bias_add_inplace(float *Y, const float *bias, int rows, int cols) {
#if defined(__AVX512F__)
  for (int i = 0; i < rows; i++) {
    float *row = Y + static_cast<size_t>(i) * cols;
    int j = 0;
    for (; j + 15 < cols; j += 16) {
      __m512 vy = _mm512_loadu_ps(row + j);
      __m512 vb = _mm512_loadu_ps(bias + j);
      vy = _mm512_add_ps(vy, vb);
      _mm512_storeu_ps(row + j, vy);
    }
    for (; j < cols; j++) row[j] += bias[j];
  }
#else
  for (int i = 0; i < rows; i++) {
    float *row = Y + static_cast<size_t>(i) * cols;
    for (int j = 0; j < cols; j++) row[j] += bias[j];
  }
#endif
}

}  // namespace

MlpInference::MlpInference() : rng_(std::random_device{}()) {
  AllocateBuffers();
  InitWeights();
  InitScalerDefaults();
  initialized_ = true;
}

MlpInference::~MlpInference() { FreeBuffers(); }

void MlpInference::AllocateBuffers() {
  W1_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kHidden1 * kInputDim));
  W2_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kHidden2 * kHidden1));
  W3_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kOutputDim * kHidden2));
  b1_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kHidden1));
  b2_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kHidden2));
  b3_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kOutputDim));
  X_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kBatch * kInputDim));
  Y1_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kBatch * kHidden1));
  Y2_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kBatch * kHidden2));
  Y3_ = static_cast<float *>(aligned_malloc64(sizeof(float) * kBatch * kOutputDim));
  assert(W1_ && W2_ && W3_ && b1_ && b2_ && b3_ && X_ && Y1_ && Y2_ && Y3_);
}

void MlpInference::InitWeights() {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  auto fill_uniform = [&](float *p, int n) {
    for (int i = 0; i < n; i++) p[i] = dist(rng_);
  };
  fill_uniform(W1_, kHidden1 * kInputDim);
  fill_uniform(W2_, kHidden2 * kHidden1);
  fill_uniform(W3_, kOutputDim * kHidden2);
  std::memset(b1_, 0, sizeof(float) * kHidden1);
  std::memset(b2_, 0, sizeof(float) * kHidden2);
  std::memset(b3_, 0, sizeof(float) * kOutputDim);
}

void MlpInference::InitScalerDefaults() {
  for (int i = 0; i < kInputDim; i++) {
    mean_[i] = 0.0f;
    scale_[i] = 1.0f;
  }
  scaler_loaded_ = false;
}

void MlpInference::FreeBuffers() {
  aligned_free(W1_);
  aligned_free(W2_);
  aligned_free(W3_);
  aligned_free(b1_);
  aligned_free(b2_);
  aligned_free(b3_);
  aligned_free(X_);
  aligned_free(Y1_);
  aligned_free(Y2_);
  aligned_free(Y3_);
}

bool MlpInference::LoadVector(const std::string &path, float *dst, size_t count, std::string *err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (err) *err = "Failed to open " + path;
    return false;
  }
  f.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(count * sizeof(float)));
  if (f.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
    if (err) *err = "Short read for " + path;
    return false;
  }
  return true;
}

bool MlpInference::LoadFromDir(const std::string &dir, std::string *err) {
  const std::string w1 = dir + "/W1.bin";
  const std::string w2 = dir + "/W2.bin";
  const std::string w3 = dir + "/W3.bin";
  const std::string b1 = dir + "/b1.bin";
  const std::string b2 = dir + "/b2.bin";
  const std::string b3 = dir + "/b3.bin";
  const std::string mean = dir + "/mean.bin";
  const std::string scale = dir + "/scale.bin";

  auto maybe_set_err = [&](const std::string &msg) {
    if (err) *err = msg;
  };

  if (!LoadVector(w1, W1_, kHidden1 * kInputDim, err)) {
    maybe_set_err("W1 load failed: " + (err ? *err : ""));
    return false;
  }
  if (!LoadVector(w2, W2_, kHidden2 * kHidden1, err)) {
    maybe_set_err("W2 load failed: " + (err ? *err : ""));
    return false;
  }
  if (!LoadVector(w3, W3_, kOutputDim * kHidden2, err)) {
    maybe_set_err("W3 load failed: " + (err ? *err : ""));
    return false;
  }
  if (!LoadVector(b1, b1_, kHidden1, err)) {
    maybe_set_err("b1 load failed: " + (err ? *err : ""));
    return false;
  }
  if (!LoadVector(b2, b2_, kHidden2, err)) {
    maybe_set_err("b2 load failed: " + (err ? *err : ""));
    return false;
  }
  if (!LoadVector(b3, b3_, kOutputDim, err)) {
    maybe_set_err("b3 load failed: " + (err ? *err : ""));
    return false;
  }

  // Scaler is optional but strongly recommended. Fall back to defaults on error.
  if (LoadVector(mean, mean_, kInputDim, nullptr) && LoadVector(scale, scale_, kInputDim, nullptr)) {
    scaler_loaded_ = true;
  } else {
    InitScalerDefaults();
  }

  weights_loaded_ = true;
  return true;
}

void MlpInference::NormalizeFeature(const float *raw, float *dst) const {
  for (int i = 0; i < kInputDim; i++) {
    const float v = raw[kFeaturePermute[i]];
    const float s = (scale_[i] == 0.0f) ? 1.0f : scale_[i];
    dst[i] = (v - mean_[i]) / s;
  }
}

void MlpInference::RunBatch(const std::vector<std::array<float, kInputDim>> &batch,
                            std::array<int, kBatch> &argmaxOut) {
  if (!initialized_) return;

  const size_t copyCount = std::min(batch.size(), static_cast<size_t>(kBatch));
  if (copyCount > 0) {
    for (size_t i = 0; i < copyCount; i++) {
      float normalized[kInputDim];
      NormalizeFeature(batch[i].data(), normalized);
      float *dst = X_ + static_cast<size_t>(i) * kInputDim;
      std::memcpy(dst, normalized, sizeof(normalized));
    }
    for (size_t i = copyCount; i < static_cast<size_t>(kBatch); i++) {
      float *dst = X_ + static_cast<size_t>(i) * kInputDim;
      std::memcpy(dst, X_ + static_cast<size_t>(copyCount - 1) * kInputDim, sizeof(float) * kInputDim);
    }
  } else {
    std::memset(X_, 0, sizeof(float) * kBatch * kInputDim);
  }

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kBatch, kHidden1, kInputDim, 1.0f,
              X_, kInputDim, W1_, kInputDim, 0.0f, Y1_, kHidden1);
  bias_relu_inplace(Y1_, b1_, kBatch, kHidden1);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kBatch, kHidden2, kHidden1, 1.0f,
              Y1_, kHidden1, W2_, kHidden1, 0.0f, Y2_, kHidden2);
  bias_relu_inplace(Y2_, b2_, kBatch, kHidden2);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kBatch, kOutputDim, kHidden2, 1.0f,
              Y2_, kHidden2, W3_, kHidden2, 0.0f, Y3_, kOutputDim);
  bias_add_inplace(Y3_, b3_, kBatch, kOutputDim);

  for (int c = 0; c < kBatch; c++) {
    const float *row = Y3_ + static_cast<size_t>(c) * kOutputDim;
    float maxVal = row[0];
    int argMax = 0;
    for (int r = 1; r < kOutputDim; r++) {
      if (row[r] > maxVal) {
        maxVal = row[r];
        argMax = r;
      }
    }
    argmaxOut[static_cast<size_t>(c)] = argMax;
  }
}

void MlpInference::RunBatch(const std::vector<std::array<uint64_t, kInputDim>> &batch,
                            std::array<int, kBatch> &argmaxOut) {
  // Convert to float features expected by the model; keep original storage as uint64_t.
  std::vector<std::array<float, kInputDim>> batchFloat;
  batchFloat.reserve(batch.size());
  for (const auto &src : batch) {
    batchFloat.push_back({static_cast<float>(src[0]),
                          static_cast<float>(src[1]),
                          static_cast<float>(src[2]),
                          static_cast<float>(src[3]),
                          static_cast<float>(src[4]),
                          static_cast<float>(src[5])});
  }
  RunBatch(batchFloat, argmaxOut);
}
