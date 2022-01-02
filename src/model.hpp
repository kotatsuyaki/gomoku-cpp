#pragma once

#include <torch/torch.h>

namespace nn = torch::nn;
using Tensor = torch::Tensor;

class Net : public nn::Module {
  public:
    Net();
    // value, policy
    std::pair<Tensor, Tensor> forward(Tensor x);

  private:
    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::Conv2d conv_v;
    nn::Linear lin;
    nn::Flatten flat;
};
