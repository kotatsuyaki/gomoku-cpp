#pragma once

#include <torch/torch.h>

namespace nn = torch::nn;
using Tensor = torch::Tensor;

class NetImpl : public nn::Module {
  public:
    NetImpl();
    // value, policy
    Tensor forward(Tensor x);
    // The user MUST call this with data on CPU
    void dump_parameters();

  private:
    nn::Conv2d conv1, conv2;
    nn::Flatten flat;

    void dump(Tensor x, std::string name);
};

TORCH_MODULE(Net);
