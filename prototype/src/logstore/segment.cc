#include <cstring>
#include <iostream>
#include <algorithm>
#include "src/logstore/manager.h"

Segment::Segment(uint64_t id, int temperature, uint64_t timestamp) {
  mSegmentId = id;
  mCreationTimestamp = timestamp;
  mClassNum = temperature;

  mMutex = new std::mutex();
  mBlocks = new uint32_t[kSegmentBlocks];
  mCategories = new int32_t[kSegmentBlocks];
  std::fill_n(mCategories, kSegmentBlocks, -1);
}

uint64_t Segment::Append(uint32_t blockAddr, int category) {
  uint64_t phyAddr = mSegmentId * kSegmentBlocks + mNextOffset;

  mBlocks[mNextOffset] = blockAddr;
  mCategories[mNextOffset] = category;
  mNextOffset += 1;
  return phyAddr;
}

void Segment::Invalidate(int i) {
  mTotalInvalidBlocks += 1;
  mBlocks[i] = UINT32_MAX;
}

uint64_t Segment::GetSegmentId() const {
  return mSegmentId;
}

bool Segment::IsFull() {
  return mNextOffset == kSegmentBlocks;
}

void Segment::Seal() {
  mSealed = true;
}

bool Segment::IsSealed() {
  return mSealed;
}

double Segment::GetGp() const {
  return 1.0 * mTotalInvalidBlocks / mNextOffset;
}

off64_t Segment::GetBlockAddr(int i) {
  return this->mBlocks[i];
}

char* Segment::GetBlockData(int i) {
  return mData + i * 4096;
}

int Segment::GetCategory(int i) const {
  if (mCategories == nullptr) return -1;
  if (i < 0 || i >= static_cast<int>(kSegmentBlocks)) return -1;
  return mCategories[i];
}

void Segment::SetCategory(int i, int category) {
  if (mCategories == nullptr) return;
  if (i < 0 || i >= static_cast<int>(kSegmentBlocks)) return;
  mCategories[i] = category;
}

// only obtain the information of the given segment
// used for selection
Segment::Segment(Segment *o) {
  this->mSegmentId = o->mSegmentId;
  this->mTotalInvalidBlocks = o->mTotalInvalidBlocks;
  this->mNextOffset = o->mNextOffset;
  this->mCreationTimestamp = o->mCreationTimestamp;
  this->mBlocks = nullptr;
  this->mCategories = nullptr;
  this->mClassNum = o->mClassNum;
}

// obtain the metadata of the block information
Segment::Segment(std::shared_ptr<Segment> o) {
  this->mSegmentId = o->mSegmentId;
  this->mTotalInvalidBlocks = o->mTotalInvalidBlocks;
  this->mNextOffset = o->mNextOffset;
  this->mCreationTimestamp = o->mCreationTimestamp;
  this->mBlocks = new uint32_t[kSegmentBlocks];
  memcpy(this->mBlocks, o->mBlocks, sizeof(uint32_t) * kSegmentBlocks);
  this->mCategories = new int32_t[kSegmentBlocks];
  std::copy(o->mCategories, o->mCategories + kSegmentBlocks, this->mCategories);
  if (this->mTotalInvalidBlocks == kSegmentBlocks) {
    this->mData = nullptr;
  } else {
    this->mData = (char*)aligned_alloc(512, kSegmentBlocks * 4096);
  }
  this->mClassNum = o->mClassNum;
}

off64_t Segment::GetPhyAddr(int i) {
  return mSegmentId * kSegmentBlocks + i;
}

char *Segment::GetData() {
  return mData;
}

uint64_t Segment::GetTotalValidBlocks() const {
  return mNextOffset - mTotalInvalidBlocks;
}

uint64_t Segment::GetTotalInvalidBlocks() const {
  return mTotalInvalidBlocks;
}

uint64_t Segment::GetTotalBlocks() const {
  return mNextOffset;
}

int Segment::GetClassNum() const {
  return mClassNum;
}

uint64_t Segment::GetAge() const {
  return Manager::globalTimestamp - mCreationTimestamp;
}

uint64_t Segment::GetCreationTimestamp() const {
  return mCreationTimestamp;
}

void Segment::Lock() {
  mMutex->lock();
}

void Segment::Unlock() {
  mMutex->unlock();
}

Segment::~Segment() {
  if (mBlocks != nullptr) {
    delete[] mBlocks;
  }
  if (mCategories != nullptr) {
    delete[] mCategories;
  }
  if (mData != nullptr) {
    free(mData);
  }
  if (mMutex != nullptr) {
    delete mMutex;
  }
}
