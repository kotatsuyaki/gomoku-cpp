#pragma once

#include <torch/torch.h>

namespace nn = torch::nn;
using Tensor = torch::Tensor;

class Net : public nn::Module {
  public:
    Net();
    Tensor forward(Tensor x);

  private:
    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::Flatten flat;
};
