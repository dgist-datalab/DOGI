#ifndef LOGSTORE_DOGISELECT_H
#define LOGSTORE_DOGISELECT_H

#include "src/selection/selection.h"

class DogiSelect : public Selection {
public:
    DogiSelect() = default;
    std::vector<std::pair<double, int>> Select(std::vector<Segment> segment) override;
};

#endif //LOGSTORE_DOGISELECT_H
