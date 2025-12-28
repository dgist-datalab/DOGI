#ifndef LOGSTORE_PLACEMENT_H
#define LOGSTORE_PLACEMENT_H

#include "src/logstore/segment.h"
class Placement {
public:
    // For GC, 'category' carries per-block category on first GC (>=0), or -1 otherwise.
    virtual int  Classify(uint32_t, bool, uint64_t, uint32_t, int category) = 0;
    virtual void Append(uint32_t addr, uint64_t timestamp) = 0;
    virtual void GcAppend(uint32_t addr) = 0;
    virtual void CollectSegment(Segment *segment) = 0;
};

#endif //LOGSTORE_PLACEMENT_H
