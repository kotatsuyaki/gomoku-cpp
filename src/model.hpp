#pragma once

#include <torch/torch.h>

namespace nn = torch::nn;
using Tensor = torch::Tensor;

class NetImpl : public nn::Module {
  public:
    NetImpl();
    // value, policy
    std::pair<Tensor, Tensor> forward(Tensor x);
    // The user MUST call this with data on CPU
    void dump_parameters();

  private:
    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::Conv2d conv_v;
    nn::Linear lin;
    nn::Flatten flat;

    void dump(Tensor x, std::string name);
};

TORCH_MODULE(Net);
