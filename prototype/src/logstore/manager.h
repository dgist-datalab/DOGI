#ifndef LOGSTORE_SEGMENT_MANAGER_H
#define LOGSTORE_SEGMENT_MANAGER_H

#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <list>
#include <vector>
#include <sys/time.h>

#include "src/logstore/segment.h"
#include "src/indexmap/indexmap.h"
#include "src/placement/placement.h"
#include "src/storage_adapter/storage_adapter.h"
#include "app/global.h"

class Manager {

public:
    static uint64_t globalTimestamp;
    Manager(int numOpenSegments);
    ~Manager();

    // group <0: let placement decide; otherwise force class id
    void Append(const void *buf, off64_t addr, int group = -1);
    bool GcAppend(const void *buf, uint32_t blockAddr, off64_t oldPhyAddr);
    void Read(void *buf, off64_t addr);

    double GetGp() const;
    uint64_t GetnBlocks() const;
    uint64_t GetnValidBlocks() const;
    uint64_t GetnInvalidBlocks() const;
    uint64_t PrintRealStats();

    void OpenNewSegment(int id);
    void RemoveSegment(int id, uint64_t nRewriteBlocks);
    void CollectSegment(int id);
    uint32_t NewClassNum(uint32_t id);

    void GetSegments(std::vector<Segment> &segs);
    Segment ReadSegment(int id);
    uint64_t GetSegmentAge(uint32_t blockAddr);

private:
    std::shared_ptr<Segment> findSegment(off64_t addr);

    std::unique_ptr<IndexMap> mIndexMap;
    std::unique_ptr<Placement> mPlacement;
    std::unordered_map<uint64_t, std::shared_ptr<Segment>> mSegments;
    std::vector<std::shared_ptr<Segment>> mOpenSegments;

    uint64_t mCurrentSegmentId{};

    uint64_t mTotalBlocks = 0;
    uint64_t mTotalInvalidBlocks = 0;
    
	uint64_t mTotalUserWrites;
    uint64_t mTotalGcWrites;
    uint64_t mTotalUserWritesG0;
    
	uint64_t mTmpUserWrites;
    uint64_t mTmpGcWrites;
    uint64_t mTmpUserWritesG0;
	uint64_t mTmpPrevTime;
	uint64_t mStartTime;
	double ClassValidBlock[10] = {0.,};
	double ClassAgeSum[10] = {0.,};
	uint64_t ClassEraseNum[10] = {0,};
	uint32_t ClassSegmentNum[10] = {0,};

    std::mutex mGlobalMutex;
    std::mutex mStopTheWorldMutex;
    std::mutex mSegmentMutex;
    std::condition_variable mStopTheWorldCv;

    std::unique_ptr<StorageAdapter> mStorageAdapter;
};


#endif //LOGSTORE_SEGMENT_MANAGER_H
