#ifndef LOGSTORE_SEGMENT_H
#define LOGSTORE_SEGMENT_H


#include <memory>
#include <mutex>

class Segment {
public:
    Segment(uint64_t id, int temperature, uint64_t timestamp);
    Segment(Segment *);
    Segment(std::shared_ptr<Segment>);
    uint64_t Append(uint32_t blockAddr, int category);
    void      Invalidate(int i);
    void      Seal();

    bool      IsFull();
    bool      IsSealed();
    char*    GetData();
    char*    GetBlockData(int i);

    off64_t  GetBlockAddr(int i);
    off64_t  GetPhyAddr(int i);

    uint64_t GetSegmentId() const;
    int      GetClassNum() const;

    uint64_t GetTotalInvalidBlocks() const;
    uint64_t GetTotalValidBlocks() const;
    uint64_t GetTotalBlocks() const;

    double   GetGp() const;
    uint64_t GetAge() const;
    uint64_t GetCreationTimestamp() const;
    int      GetCategory(int i) const;
    void     SetCategory(int i, int category);

    void Lock();
    void Unlock();

    ~Segment();
    uint64_t mCreationTimestamp = 0;

private:
    uint64_t mSegmentId = 0;
    bool     mSealed = false;
    int      mClassNum = 0;

    uint64_t mTotalInvalidBlocks = 0;

    uint32_t* mBlocks = nullptr; // storing the logical block addresses of the blocks
    int32_t* mCategories = nullptr; // per-block category state (-1 after first GC)
    uint64_t mNextOffset = 0; // the offset of next appended block

    char *mData = nullptr; // storing the data of the blocks, only used when GC

    std::mutex* mMutex = nullptr;
};


#endif //LOGSTORE_SEGMENT_H
