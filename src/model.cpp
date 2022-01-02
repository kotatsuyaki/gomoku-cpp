#include "model.hpp"

#include <ATen/Functions.h>
#include <algorithm>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <torch/nn/modules/linear.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/padding.h>

Net::Net()
    : conv1(
          nn::Conv2dOptions(1, 3, 3).stride(1).in_channels(1).out_channels(32)),
      conv2(nn::Conv2dOptions(1, 3, 3).stride(1).in_channels(32).out_channels(
          32)),
      conv3(nn::Conv2dOptions(1, 3, 3).stride(1).in_channels(32).out_channels(
          32)),
      conv4(
          nn::Conv2dOptions(1, 3, 3).stride(1).in_channels(32).out_channels(1)),
      flat(nn::Flatten()) {
    flat->options.start_dim(1).end_dim(3);
    register_module("conv1", conv1);
}

Tensor Net::forward(Tensor x) {
    auto padopts = torch::nn::functional::PadFuncOptions({1, 1, 1, 1});

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv1(x));

    fmt::print("pass\n");

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv2(x));

    fmt::print("pass\n");

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv3(x));

    fmt::print("pass\n");

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv4(x));

    fmt::print("pass\n");

    x = flat(x);
    x = torch::log_softmax(x, -1);

    fmt::print("sizes = {}\n", x.sizes());

    return x;
}
