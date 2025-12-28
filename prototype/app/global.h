#ifndef GLOBALS_H
#define GLOBALS_H

#include <stdint.h>  // uint32_t, uint64_t
#include <stdio.h>   // FILE*

// Global configuration knobs shared across app components.
extern char *PlacementName;
extern double GpThreshold;
extern double OpRatio;
extern int LogicalSizeGb;
extern char wk_name[128];
extern uint32_t NumGroup;
extern uint32_t naive_start;
extern uint64_t BIR[10];
extern int APPLY_ML;

// Device paths/names for zoned backend.
extern const char kZnsDevicePath[];
extern const char kZbdDeviceName[];

// Segment/Block size constants (configured in global.cc)
extern const uint64_t kBlockBytes;
extern const uint64_t kSegmentBlocks;         // blocks per segment
extern const uint64_t kSegmentBytes;          // bytes per segment

// Derived thresholds based on logical size.
inline uint64_t GetPassTimeBlocks() {
  return static_cast<uint64_t>(LogicalSizeGb) * 1000ull * 1000ull * 1000ull / kBlockBytes;
}
#endif // GLOBALS_H
