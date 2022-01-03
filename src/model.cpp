#include "model.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <numeric>

#include <ATen/Functions.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/padding.h>

NetImpl::NetImpl()
    : conv1(nn::Conv2dOptions(1, 3, 3)
                .stride(1)
                .in_channels(1)
                .out_channels(32)
                .bias(false)),
      conv2(nn::Conv2dOptions(1, 3, 3)
                .stride(1)
                .in_channels(32)
                .out_channels(1)
                .bias(false)),
      flat(nn::Flatten()) {
    flat->options.start_dim(1).end_dim(3);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
}

void NetImpl::dump_parameters() {
    dump(conv1->weight, "conv1.weight");
    dump(conv1->bias, "conv1.bias");
    dump(conv2->weight, "conv1.weight");
    dump(conv2->bias, "conv1.bias");
}

void NetImpl::dump(Tensor x, std::string name) {
    auto fname = name + ".txt";

    fmt::print("Dumping data of {} to {}\n", name, fname);
    auto sizes = x.sizes();
    fmt::print("{} shape = {}\n", name, sizes);

    int numel =
        std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());

    std::ofstream file(fname);
    for (int i = 0; i < numel; i += 1) {
        float f = static_cast<float*>(x.data_ptr())[i];
        fmt::print(file, "{}\n", f);
    }
    file.close();
}

// value, policy
Tensor NetImpl::forward(Tensor x) {
    auto padopts = torch::nn::functional::PadFuncOptions({1, 1, 1, 1});

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv1(x));

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv2(x));

    x = flat(x);
    x = torch::log_softmax(x, -1);

    return x;
}
