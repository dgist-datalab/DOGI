#include "app/global.h"

char *PlacementName = nullptr;
double GpThreshold = 0.091;               // for GC (how many invalid pages)
double OpRatio     = 0.10;                // over-provisioning ratio (physical = logical * (1 + OpRatio))
int LogicalSizeGb  = 8;                   // logical workload size in GB
char wk_name[128]  = "/home/dogi-ae/dogi-prototype/prototype/app/workload/test-fio-small";
//char wk_name[128]  = "/test-fio-small";
uint32_t NumGroup  = 6;
uint64_t BIR[10]   = {0,};
uint32_t naive_start = 1;
int APPLY_ML = 0;

const char kZnsDevicePath[] = "/dev/nvme0n1";
const char kZbdDeviceName[] = "nvme0n1";

const uint64_t kBlockBytes    = 4096;
const uint64_t kSegmentBlocks = 16384; // blocks per segment
const uint64_t kSegmentBytes  = kBlockBytes * kSegmentBlocks;
