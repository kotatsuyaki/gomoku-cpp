#pragma once

#include <torch/torch.h>

namespace nn = torch::nn;
using Tensor = torch::Tensor;

class NetImpl : public nn::Module {
  public:
    NetImpl();
    // value, policy
    Tensor forward(Tensor x, bool print = false);
    // The user MUST call this with data on CPU
    void dump_parameters();
    Tensor manual_forward(Tensor x);

  private:
    nn::Conv2d conv1, conv2, conv3;
    nn::Flatten flat;
};

TORCH_MODULE(Net);
