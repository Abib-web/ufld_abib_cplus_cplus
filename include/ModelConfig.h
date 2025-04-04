#pragma once
#include <vector>
#include "ModelType.h"

struct ModelConfig {
    int img_w;
    int img_h;
    std::vector<int> row_anchor;
    int griding_num;
    int cls_num_per_lane;

    ModelConfig(ModelType model_type);
};