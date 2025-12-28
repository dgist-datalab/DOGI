#ifndef LOGSTORE_BLOCKDEVICE_H
#define LOGSTORE_BLOCKDEVICE_H


#include <memory>
#include "src/logstore/manager.h"
#include "src/logstore/scheduler.h"
#include "src/buse/buseOperations.h"

class LogStore : public buse::buseOperations {

public:
    LogStore(uint64_t, uint32_t);
    ~LogStore() { Shutdown(); }

    int read(void *buf, size_t len, off64_t offset);
    // group <0 means auto placement classification
    int write(const void *buf, size_t len, off64_t offset, int group = -1);
    uint64_t GetSegmentAge(uint32_t blockAddr) { return mManager->GetSegmentAge(blockAddr); }

    void Shutdown();

private:
    std::unique_ptr<Manager> mManager;
    std::unique_ptr<Scheduler> mScheduler;
};

#endif //LOGSTORE_BLOCKDEVICE_H
