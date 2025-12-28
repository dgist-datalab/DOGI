#include "src/placement/dogi.h"
#include "app/classifier.h"
#include "app/global.h"
#include "app/group_optimizer.h"

DOGI::DOGI() {}

int DOGI::Classify(uint32_t blockAddr, bool isGcAppend, uint64_t /*Age*/, uint32_t PrevClass, int category) {
  static uint64_t fgPrintCounter = 0;
  static uint64_t gcPrintCounter = 0;
  constexpr uint64_t kPrintEvery = 1000; // print every N classifications when APPLY_ML is on

  if (!isGcAppend) {
    int mapped = MapCategoryToGroup(category); // user-provided category
    if (mapped < 0) mapped = 0;
    if (mapped >= static_cast<int>(NumGroup)) mapped = static_cast<int>(NumGroup) - 1;
    if (APPLY_ML && (++fgPrintCounter % kPrintEvery == 0)) {
      //printf("category: %d, Group: %d\n", category, mapped){
    }
    return mapped;
  }

  uint64_t filterValue = FrozenFilterManager::Instance().Query(blockAddr);
  if (filterValue == (uint64_t)0) {
    // frozen -> send to frozen group (original behavior)
    return NumGroup - 1;
  }

  // 'category' carries category on first GC; otherwise -1 for later GCs
  int next = MapCategoryFirstGc(category, static_cast<int>(PrevClass));
  if (APPLY_ML && (++gcPrintCounter % kPrintEvery == 0)) {
    //printf("GC category: %d, PrevGroup: %u -> NextGroup: %d\n", category, PrevClass, next);
  }
  //if (next < 0) next = PrevClass;
  //if (next >= static_cast<int>(NumGroup)) next = static_cast<int>(NumGroup) - 1;
  return next;
}

void DOGI::Append(uint32_t blockAddr, uint64_t timestamp) {
}

void DOGI::GcAppend(uint32_t blockAddr) {
}

void DOGI::CollectSegment(Segment *segment) {
}
