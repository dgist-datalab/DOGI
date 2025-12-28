#ifndef GLOBALS_H
#define GLOBALS_H

#include <stdint.h>  // uint32_t, uint64_t
#include <stdio.h>   // FILE*

// Global configuration knobs shared across app components.
extern uint64_t SwWp;
extern char *PlacementName;
extern double GpThreshold;
extern double OpRatio;
extern int LogicalSizeGb;
extern uint32_t NumGroup;
extern uint32_t naive_start;
extern uint64_t BIR[10];
extern int APPLY_ML;

// Segment/Block size constants
inline constexpr uint64_t kBlockBytes = 4096;
inline constexpr uint64_t kSegmentBlocks = 16384;           // blocks per segment
//inline constexpr uint64_t kSegmentBlocks = 65536;           // blocks per segment
inline constexpr uint64_t kSegmentBytes  = kBlockBytes * kSegmentBlocks;
#endif // GLOBALS_H
