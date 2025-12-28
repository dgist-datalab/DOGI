#ifndef LOGSTORE_INDEXMAPFACTORY_H
#define LOGSTORE_INDEXMAPFACTORY_H

#include "src/indexmap/indexmap.h"
#include "src/indexmap/hashmap.h"
#include "src/indexmap/array.h"
#include "src/indexmap/filter.h"
#include <string>
#include <iostream>

class IndexMapFactory {
  public:
    static IndexMap *GetInstance(std::string type) {
      if (type == "HashMap") {
        return new HashMap();
      } else if (type == "Array") {
        return new Array();
      } else if (type == "Filter") {
		return new Filter();
	  } else {
        std::cerr << "No IndexMap, type: " << type << std::endl;
      }
      return new HashMap();
    }
};

#endif
