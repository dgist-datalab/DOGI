
#ifndef LOGSTORE_PLACEMENT_FACTORY_H
#define LOGSTORE_PLACEMENT_FACTORY_H

#include "placement.h"
#include "dogi.h"
#include <string>
#include <iostream>

class PlacementFactory {
  public:
    static Placement *GetInstance(std::string type) {
      std::cout << "Placement algorithm: " << type << std::endl;
      (void)type;
      return new DOGI();
    }
};

#endif
