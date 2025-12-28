#ifndef LOGSTORE_DOGI_H
#define LOGSTORE_DOGI_H

#include "src/logstore/segment.h"
#include "src/placement/placement.h"

class DOGI : public Placement {
public:
   DOGI();
   int  Classify(uint32_t blockAddr, bool isGcAppend, uint64_t Age, uint32_t PrevClass, int category) override;
   void Append(uint32_t blockAddr, uint64_t timestamp) override;
   void GcAppend(uint32_t blockAddr) override;
   void CollectSegment(Segment *segment) override;
};

#endif //LOGSTORE_DOGI_H
