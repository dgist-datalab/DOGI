#ifndef LOGSTORE_FILTER_H
#define LOGSTORE_FILTER_H

#include "indexmap.h"

class Filter : public IndexMap {
public:
  Filter(int capacity);
  Filter();
  ~Filter();
  uint64_t Query(uint32_t blockAddr) override;
  void Update(uint32_t blockAddr, uint64_t type) override;

private:
  int mCapacity = 128 * 1024 * 1024; // 512 GiB
  uint64_t *mValues;
};

#endif //LOGSTORE_FILTER_H
