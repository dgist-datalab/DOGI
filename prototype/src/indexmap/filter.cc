#include "src/indexmap/filter.h"
#include <iostream>

Filter::Filter(int capacity) {
  mCapacity = capacity;
  mValues = new uint64_t[mCapacity];
  for (int i = 0; i < mCapacity; ++i) {
    mValues[i] = 0;
  }
}

Filter::Filter() {
  mValues = new uint64_t[mCapacity];
  for (int i = 0; i < mCapacity; ++i) {
    mValues[i] = 0;
  }
}

void Filter::Update(uint32_t blockAddr, uint64_t type) {
	mValues[blockAddr] = type;
}

uint64_t Filter::Query(uint32_t blockAddr) {
  return mValues[blockAddr];
}

Filter::~Filter() {
  delete[] mValues;
}
