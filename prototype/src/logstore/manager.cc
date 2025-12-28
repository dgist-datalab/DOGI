#include <cassert>
#include <mutex>
#include <iostream>
#include <thread>

#include "src/logstore/config.h"
#include "src/logstore/manager.h"
#include "src/indexmap/factory.h"
#include "src/storage_adapter/factory.h"
#include "src/placement/factory.h"
#include "app/global.h"
#include "app/classifier.h"
#include "app/group_optimizer.h"

uint64_t Manager::globalTimestamp = 0;
Manager::Manager(int numOpenSegments) {
  mIndexMap = std::unique_ptr<IndexMap>(IndexMapFactory::GetInstance(Config::GetInstance().indexMap));
  mPlacement = std::unique_ptr<Placement>(PlacementFactory::GetInstance(Config::GetInstance().placement));
  mStorageAdapter = std::unique_ptr<StorageAdapter>(StorageAdapterFactory::GetInstance(Config::GetInstance().storageAdapter));

  std::lock_guard<std::mutex> lck(mSegmentMutex);
  for (int i = 0; i < numOpenSegments; ++i) {
    std::shared_ptr<Segment> segment = std::make_shared<Segment>(mCurrentSegmentId, i, globalTimestamp);
    mOpenSegments.emplace_back(segment);
    mSegments[segment->GetSegmentId()] = std::move(segment);
    mStorageAdapter->CreateSegment(mCurrentSegmentId);
    mCurrentSegmentId += 1;
  }
  struct timeval current_time;
  gettimeofday(&current_time, NULL);
  mStartTime = (uint64_t)current_time.tv_sec; 
  mTmpPrevTime = (uint64_t)0;

  // Initialize counters
  mTotalUserWrites = 0;
  mTotalGcWrites = 0;
  mTotalUserWritesG0 = 0;
  mTmpUserWrites = 0;
  mTmpGcWrites = 0;
  mTmpUserWritesG0 = 0;
}

Manager::~Manager() {
  std::cout << "UserWrite: " << mTotalUserWrites * 4096 / 1024 / 1024 / 1024.0 << 
    ", GCWrite: " << mTotalGcWrites * 4096 / 1024 / 1024 / 1024.0 << std::endl;
}

/*
 * the data must be length of 4KiB
 * logical addr: in units of bytes
 * physical addr: in units of 4KiB blocks
 */
void Manager::Append(const void *buf, off64_t addr, int group) {
  using namespace std::chrono_literals;
//  if (GetGp() > 0.1737) {
  if (GetGp() > GpThreshold) {
    std::this_thread::sleep_for(0.0001s);
  }

  uint32_t blockAddr;
  int classId;
  uint64_t oldPhyAddr, newPhyAddr;
  uint64_t PrevAge = 0;
  uint64_t PrevClass = 6;

  mGlobalMutex.lock();
  blockAddr = addr / 4096;
  
  oldPhyAddr = mIndexMap->Query(blockAddr);
  if (oldPhyAddr != ~0ull) {
    std::shared_ptr<Segment> oldSegment = findSegment(oldPhyAddr);
    oldSegment->Invalidate(oldPhyAddr % kSegmentBlocks);
    if (oldSegment->IsSealed()) {
      mTotalInvalidBlocks += 1;
    }
	PrevAge = oldSegment->GetAge();
	PrevClass = oldSegment->GetClassNum();
  }
  // 'group' is treated as category (0=hot, 1..N=ML output)
  classId = mPlacement->Classify(blockAddr, false, (uint64_t)PrevAge, (uint32_t)PrevClass, /*category=*/static_cast<int>(group));

  std::shared_ptr<Segment> currentSegment = mOpenSegments[classId];
  newPhyAddr = currentSegment->Append(blockAddr, /*category=*/static_cast<int>(group));
  mIndexMap->Update(blockAddr, newPhyAddr);

  mPlacement->Append(blockAddr, globalTimestamp);
  globalTimestamp += 1; // tick inced by one
  mTotalUserWrites += 1;
  mTmpUserWrites += 1;
  if (group == 0) {
    mTotalUserWritesG0 += 1;
    mTmpUserWritesG0 += 1;
  }

  if (currentSegment->IsFull()) {
    currentSegment->Seal();
    mTotalInvalidBlocks += currentSegment->GetTotalInvalidBlocks();
    mTotalBlocks += kSegmentBlocks;	
    //if (strcmp(PlacementName, "DOGI")==0) currentSegment->mCreationTimestamp = globalTimestamp;
    OpenNewSegment(classId);
  }

  Config::GetInstance().numValidBlocks = GetnValidBlocks();
  mGlobalMutex.unlock();

  currentSegment->Lock();
  mStorageAdapter->Write(buf, currentSegment->GetSegmentId(), newPhyAddr % kSegmentBlocks); // data, segmentId, offset
  currentSegment->Unlock();
  if (globalTimestamp % 5242880 == 0) PrintRealStats();
}

bool Manager::GcAppend(const void *buf, uint32_t blockAddr, off64_t oldPhyAddr) {
  mGlobalMutex.lock();
  int classId;
  uint64_t CurrAge;
  uint64_t CurrClass;
  std::shared_ptr<Segment> oldSegment = findSegment(oldPhyAddr);
  uint64_t phyAddr = mIndexMap->Query(blockAddr);
  if (oldPhyAddr != phyAddr) {
    mGlobalMutex.unlock();
    return false;
  }

  CurrAge = oldSegment->GetAge();
  CurrClass = oldSegment->GetClassNum();
  int catState = oldSegment->GetCategory(oldPhyAddr % kSegmentBlocks);
  classId = mPlacement->Classify(blockAddr, true, (uint64_t)CurrAge, (uint32_t)CurrClass, /*category=*/catState);
  HotIntervalTracker::Instance().AccumulateGcAge(blockAddr, CurrAge);

  std::shared_ptr<Segment> currentSegment = mOpenSegments[classId];
  phyAddr = currentSegment->Append(blockAddr, /*category=*/catState >= 0 ? -1 : -1);

  mIndexMap->Update(blockAddr, phyAddr);
  mPlacement->GcAppend(blockAddr);
  mTotalGcWrites += 1;
  mTmpGcWrites += 1;

  if (currentSegment->IsFull()) {
    currentSegment->Seal();
    mTotalInvalidBlocks += currentSegment->GetTotalInvalidBlocks();
    mTotalBlocks += kSegmentBlocks;
    OpenNewSegment(classId);
  }
  mGlobalMutex.unlock();

  currentSegment->Lock();
  mStorageAdapter->Write(buf, currentSegment->GetSegmentId(), phyAddr % kSegmentBlocks);
  currentSegment->Unlock();

  return true;
}

void Manager::Read(void *buf, off64_t addr) {
  uint32_t blockAddr = addr / 4096;
  off64_t phyAddr = mIndexMap->Query(blockAddr);
  if (phyAddr == ~0ull) {
    return;
  }

  std::shared_ptr<Segment> segment = findSegment(phyAddr);
  // read data from segmentId with offset (phyAddr % kSegmentBlocks) and store it in buf
  mStorageAdapter->Read(buf, segment->GetSegmentId(), phyAddr % kSegmentBlocks);
}

Segment Manager::ReadSegment(int id) {
  Segment segment = Segment(mSegments[id]);
  if (segment.GetTotalValidBlocks() != 0) {
    for (uint64_t i = 0; i < kSegmentBlocks; ++i) {
      off64_t blockAddr = segment.GetBlockAddr(i);
      if (blockAddr == ~0ull) continue;
      off64_t phyAddr = segment.GetPhyAddr(i);
      char* data = segment.GetBlockData(i);
      mStorageAdapter->Read(data, id, phyAddr % kSegmentBlocks);
    }
  }
  return segment;
}

uint32_t Manager::NewClassNum(uint32_t id) {
    /*
	if (strcmp(PlacementName, "DOGI") == 0){
        if (id == 0) return 0;
        else if (id == 1) return 1;
        else if (id == 2) return 1;
        else return id-1;
    }
    else {
        return id;
    }
	*/
    return id;
}

void Manager::OpenNewSegment(int id) {
  std::lock_guard<std::mutex> lck(mSegmentMutex);
  uint32_t ClassId = NewClassNum((uint32_t)id);
  std::shared_ptr<Segment> segment = std::make_shared<Segment>(mCurrentSegmentId, id, globalTimestamp);
  mOpenSegments[id] = segment;
  mSegments[segment->GetSegmentId()] = std::move(segment);
  mStorageAdapter->CreateSegment(mCurrentSegmentId);
  mCurrentSegmentId += 1;
  ClassSegmentNum[ClassId] += 1;
}

void Manager::RemoveSegment(int id, uint64_t nInvalidBlocks) {
  {

	uint32_t ClassId = mSegments[id]->GetClassNum();
	ClassId = NewClassNum(ClassId);
	std::lock_guard<std::mutex> lck(mSegmentMutex);
    mTotalInvalidBlocks -= mSegments[id]->GetTotalInvalidBlocks();
    ClassValidBlock[ClassId] += (static_cast<double>(kSegmentBlocks) - mSegments[id]->GetTotalInvalidBlocks())/static_cast<double>(kSegmentBlocks);
	ClassEraseNum[ClassId] += 1;
	ClassSegmentNum[ClassId] -= 1; 
    ClassAgeSum[ClassId] += (double)(mSegments[id]->GetAge());
    mTotalBlocks -= kSegmentBlocks;
		
    mSegments.erase(id);

  }
	
  mStorageAdapter->DestroySegment(id);
//  PrintRealStats();
}

void Manager::CollectSegment(int id) {
  std::lock_guard<std::mutex> lck(mSegmentMutex);
  mPlacement->CollectSegment(mSegments[id].get());
}

double Manager::GetGp() const {
  return (mTotalBlocks == 0) ? 0 : 1.0 * mTotalInvalidBlocks / mTotalBlocks;
}

void Manager::GetSegments(std::vector<Segment> &segs) {
  std::lock_guard<std::mutex> lck(mSegmentMutex);
  segs.reserve(mSegments.size());
  for (auto pr : mSegments) {
    Segment *segment = pr.second.get();
    Segment tmp(segment);
    if (segment->IsSealed()) {
      segs.emplace_back(tmp);
    }
  }
}

uint64_t Manager::GetnValidBlocks() const {
  return mTotalBlocks - mTotalInvalidBlocks;
}

uint64_t Manager::GetnBlocks() const {
  return mTotalBlocks;
}

uint64_t Manager::GetSegmentAge(uint32_t blockAddr) {
  std::lock_guard<std::mutex> lck(mGlobalMutex);
  off64_t phyAddr = mIndexMap->Query(blockAddr);
  if (phyAddr == ~0ull) return 0;
  std::shared_ptr<Segment> seg = findSegment(phyAddr);
  return seg ? seg->GetAge() : 0;
}

uint64_t Manager::GetnInvalidBlocks() const {
  return mTotalInvalidBlocks;
}

// Stats are printed per 20GiB window (5,242,880 blocks); align with the trigger in Append if you retune the interval.
uint64_t Manager::PrintRealStats() {

  uint64_t nInvalidBlocks = 0, nBlocks = 0;
  struct timeval current_time;
  for (auto pr : mSegments) {
    nInvalidBlocks += pr.second->GetTotalInvalidBlocks();
    nBlocks += pr.second->GetTotalBlocks();
  }
  gettimeofday(&current_time, NULL);
  
  double mTmpThroughput = (20*1024*1024*1024.)/((uint64_t)current_time.tv_sec - mTmpPrevTime);
  double Throughput = mTotalUserWrites * 4096. / ((uint64_t)current_time.tv_sec - mStartTime);

  std::cout << "============== Per 20GiB Info ==============" << std::endl;
  std::cout << "GlobalTimeStamp:" << globalTimestamp << std::endl;
  printf("Per20GiB Throughput: %.3f MiB/s, Total Throughput: %.3f MiB\n", mTmpThroughput/1024/1024., Throughput/1024./1024.);
  printf("UserWrite: %.1f GiB\n", mTotalUserWrites * 4096 / 1024 / 1024 / 1024.0); 
  printf("GCWrite: %.1f GiB\n", mTotalGcWrites * 4096 / 1024 / 1024 / 1024.0); 
  printf("Per20GiB WAF: %.3f, Total WAF: %.3f\n", (double)(mTmpUserWrites+mTmpGcWrites)/(mTmpUserWrites), (double)(mTotalUserWrites+mTotalGcWrites)/mTotalUserWrites);

  // HotFilter -> G0 user write share
  double per_g0_pct = (mTmpUserWrites > 0) ? (100.0 * (double)mTmpUserWritesG0 / (double)mTmpUserWrites) : 0.0;
  double total_g0_pct = (mTotalUserWrites > 0) ? (100.0 * (double)mTotalUserWritesG0 / (double)mTotalUserWrites) : 0.0;
  printf("Per20GiB HotFilter->G0 UserWrites: %lu (%.2f%%)\n", mTmpUserWritesG0, per_g0_pct);
  printf("Total   HotFilter->G0 UserWrites: %lu (%.2f%%)\n", mTotalUserWritesG0, total_g0_pct);
  for (int i = 0; i < NumGroup; i++) {
    //if (strcmp(PlacementName, "DOGI")== 0){
       // if (i == NumGroup - 1) continue;
    //}
    printf("Class %d[%d]: %.3f (Erase: %ld) (Age: %.3f)\n", i, ClassSegmentNum[i], ClassValidBlock[i]/ClassEraseNum[i], ClassEraseNum[i], ClassAgeSum[i]/ClassEraseNum[i]); 
  }
  std::cout << "============================================" << std::endl;
//  std::cout << "Stat: " << nInvalidBlocks << " " << nBlocks - nInvalidBlocks << std::endl;
  mTmpUserWrites = 0;
  mTmpGcWrites = 0;
  mTmpUserWritesG0 = 0;
  mTmpPrevTime = (uint64_t)current_time.tv_sec;
  for (int i = 0; i < 10; i++) {
	ClassValidBlock[i] = 0.;
	ClassAgeSum[i] = 0.;
	ClassEraseNum[i] = 0;
  }
  return 0;
}

std::shared_ptr<Segment> Manager::findSegment(off64_t phyAddr) {
  // the first 32bit is used as the segment id while the last 32bit is used as the offset inside the segment
  uint64_t segmentId = phyAddr / kSegmentBlocks;
  std::shared_ptr<Segment> result = mSegments[segmentId];
  assert(result->GetSegmentId() == segmentId);
  return result;
}
