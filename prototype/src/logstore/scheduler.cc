#include "src/logstore/scheduler.h"
#include "app/global.h"

Scheduler::Scheduler(Manager *manager)
{
  mSelection = std::unique_ptr<Selection>(SelectionFactory::GetInstance(Config::GetInstance().selection));
  mWorker = std::thread(&Scheduler::scheduling, this, manager);
  mWorker.detach();
}

void Scheduler::scheduling(Manager *manager) {
  using namespace std::chrono_literals;
  struct timeval current_time;
  while (true) {
    std::this_thread::sleep_for(0.05s);
    if (mShutdown) {
      break;
    }


//    while (manager->GetGp() >= 0.1737) {
    while (manager->GetGp() >= GpThreshold) {
      //printf("GP: %.2f\n", manager->GetGp());
      // select a segment
      int segmentId = select(manager);
      {
        gettimeofday(&current_time, NULL);
 //       printf("GC start: %ld.%ld\n",
 //           current_time.tv_sec, current_time.tv_usec);
      }
      Segment segment = manager->ReadSegment(segmentId);
	  {
//		printf("GC Semgent Info: ClassId: %d, Valid Ratio: %.3f\n", segment.GetClassNum(), 1. - segment.GetGp()); 
	    gettimeofday(&current_time, NULL);
//        printf("GC finish read: %ld.%ld\n",
//            current_time.tv_sec, current_time.tv_usec);
      }
      // collect the segment
      collect(manager, segment);
      {
        gettimeofday(&current_time, NULL);
//        printf("GC finish rewrite: %ld.%ld\n",
//            current_time.tv_sec, current_time.tv_usec);
      }
    }
  }
}

int Scheduler::select(Manager *manager) {
  // prepare the segments_
  std::vector<Segment> segments;
  manager->GetSegments(segments);

  auto res = mSelection->Select(segments);

  return res[0].second;
}

void Scheduler::collect(Manager *manager, Segment &segment) {
  uint64_t nRewriteBlocks = 0;
  manager->CollectSegment(segment.GetSegmentId());
  for (uint64_t i = 0; i < kSegmentBlocks; ++i) {
    off64_t blockAddr = segment.GetBlockAddr(i);
    if (blockAddr == UINT32_MAX) continue;
    off64_t oldPhyAddr = segment.GetPhyAddr(i);
    char* data = segment.GetBlockData(i);
    if (!manager->GcAppend(data, blockAddr, oldPhyAddr)) {
      nRewriteBlocks += 1;
    }
  }
  manager->RemoveSegment(segment.GetSegmentId(), segment.GetTotalInvalidBlocks() + nRewriteBlocks);
}
