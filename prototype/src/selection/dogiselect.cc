#include <cmath>
#include <algorithm>
#include <float.h>
#include <iostream>
#include <cstring>
#include <array>
#include "src/selection/dogiselect.h"
#include "src/logstore/manager.h"
#include "app/global.h"

std::vector<std::pair<double, int>> DogiSelect::Select(std::vector<Segment> segments) {
  uint64_t globalTimestamp = Manager::globalTimestamp;
  std::array<std::vector<std::pair<double, int>>, 10> vecs; // sized for up to 10 configured groups

  for (int i = 0; i < segments.size(); ++i) {
    Segment &o = segments[i];
	uint32_t ClassId = o.GetClassNum();

	uint32_t MappedClassId; //Real Group Number
	if (ClassId < naive_start) {
		MappedClassId = ClassId;
	}
	else MappedClassId = naive_start; 
	double score;
    double gp = o.GetGp();
    double age = globalTimestamp - o.GetCreationTimestamp();

	if (MappedClassId < naive_start) {
		score = age;
		if (age > BIR[MappedClassId]) {
			vecs[MappedClassId].emplace_back(score, o.GetSegmentId());
		}
	} else {
		score = (gp == 1.0) ? DBL_MAX : gp / (1 - gp) * sqrt(age);
		vecs[MappedClassId].emplace_back(score, o.GetSegmentId());
	}
  }

    for (int i = 0; i <= naive_start; i++) {
        if (!vecs[i].empty()) {
            std::sort(vecs[i].begin(), vecs[i].end(),
                      [](auto &a, auto &b) {
                          return (a.first > b.first);
                      });
            return vecs[i];
        }
    }
 
  std::cout << "WHY GC is triggered even though there is no segment for GC victim?" << std::endl;
  abort();
  return vecs[3];
}
