#include "app/global.h"

char *PlacementName = nullptr;
double GpThreshold = 0.091;               // for GC (how many invalid pages)
double OpRatio     = 0.10;                // over-provisioning ratio (physical = logical * (1 + OpRatio))
int LogicalSizeGb  = 8;                   // logical workload size in GB
char wk_name[128]  = "/home/nkgy/fio_bench/test-fio-small";
uint32_t NumGroup  = 6;
uint64_t BIR[10]   = {0,};
uint32_t naive_start = 1;
int APPLY_ML = 0;

const char kZnsDevicePath[] = "/dev/nvme2n2";
const char kZbdDeviceName[] = "nvme2n2";

const uint64_t kBlockBytes    = 4096;
const uint64_t kSegmentBlocks = 16384; // blocks per segment
const uint64_t kSegmentBytes  = kBlockBytes * kSegmentBlocks;
