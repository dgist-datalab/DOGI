#ifndef LOGSTORE_CONFIG_H
#define LOGSTORE_CONFIG_H

#include <string>
#include "app/global.h"
#include <cstring>

class Config {
public:
  static Config& GetInstance() {
    static Config instance;
    return instance;
  }
  
  std::string selection = "DogiSelect";
  std::string indexMap = "Array";
  std::string storageAdapter = "ZenFS";
  std::string placement = "DOGI";
  int         maxNumOpenSegments = 6;
  uint64_t    numValidBlocks = 0;

  // For zenfs and zoned storage backend
  std::string zbdName = kZbdDeviceName;
  std::string zenFsAuxPath = "/tmp/aux_path";

  // For local file system backend
  std::string localAdapterDir = "/tmp/local";
};

#endif
