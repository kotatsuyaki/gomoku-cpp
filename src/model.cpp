#include "model.hpp"

#include <ATen/Functions.h>
#include <algorithm>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <torch/nn/modules/linear.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/padding.h>

NetImpl::NetImpl()
    : conv1(
          nn::Conv2dOptions(1, 3, 3).stride(1).in_channels(1).out_channels(32)),
      conv2(nn::Conv2dOptions(1, 3, 3).stride(1).in_channels(32).out_channels(
          32)),
      conv3(nn::Conv2dOptions(1, 3, 3).stride(1).in_channels(32).out_channels(
          32)),
      conv4(
          nn::Conv2dOptions(1, 3, 3).stride(1).in_channels(32).out_channels(1)),
      conv_v(
          nn::Conv2dOptions(1, 3, 3).stride(1).in_channels(32).out_channels(1)),
      flat(nn::Flatten()), lin(nn::LinearOptions(36, 1)) {
    flat->options.start_dim(1).end_dim(3);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("conv_v", conv_v);
    register_module("flat", flat);
    register_module("lin", lin);
}

// value, policy
std::pair<Tensor, Tensor> NetImpl::forward(Tensor x) {
    auto padopts = torch::nn::functional::PadFuncOptions({1, 1, 1, 1});

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv1(x));

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv2(x));

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv3(x));

    auto policy = nn::functional::pad(x, padopts);
    policy = torch::relu(conv4(policy));
    policy = flat(policy);
    policy = torch::log_softmax(policy, -1);

    auto value = nn::functional::pad(x, padopts);
    value = torch::relu(conv_v(value));
    value = flat(value);
    value = torch::tanh(lin(value));

    return {value, policy};
}
