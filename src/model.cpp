#include "model.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>

#include <ATen/Functions.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/padding.h>

using namespace torch::indexing;

NetImpl::NetImpl()
    : conv1(nn::Conv2dOptions(1, 20, 3).stride(1)),
      conv2(nn::Conv2dOptions(20, 20, 3).stride(1)),
      conv3(nn::Conv2dOptions(20, 1, 3).stride(1)), flat(nn::Flatten()) {
    flat->options.start_dim(1).end_dim(3);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
}

namespace {
void dump_text(Tensor x, std::string name) {
    auto fname = name + ".txt";

    auto sizes = x.sizes();
    int numel =
        std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());

    std::ofstream file(fname);
    for (int i = 0; i < numel; i += 1) {
        float f = static_cast<float*>(x.data_ptr())[i];
        fmt::print(file, "{}\n", f);
    }
    file.close();
}

void dump_bin(Tensor x, std::string name) {
    auto fname = name + ".bin";

    auto sizes = x.sizes();
    int numel =
        std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());

    std::ofstream file(fname);
    for (int i = 0; i < numel; i += 1) {
        float f = static_cast<float*>(x.data_ptr())[i];
        int16_t quantized = std::clamp(static_cast<int16_t>(f * std::pow(2, 8)),
                                       std::numeric_limits<int16_t>::min(),
                                       std::numeric_limits<int16_t>::max());
        for (int b = 15; b >= 0; b -= 1) {
            fmt::print(file, "{}", ((quantized >> b) & 1) ? '1' : '0');
            if (b % 4 == 0 && b != 0) {
                fmt::print(file, "_");
            }
        }
        fmt::print(file, "\n");
    }
    file.close();
}

void dump(Tensor x, std::string name) {
    fmt::print("Dumping data of {} (sizes = {})\n", name, x.sizes());
    dump_text(x, name);
    dump_bin(x, name);
}
} // namespace

void NetImpl::dump_parameters() {
    dump(conv1->weight, "conv1.weight");
    dump(conv1->bias, "conv1.bias");
    dump(conv2->weight, "conv2.weight");
    dump(conv2->bias, "conv2.bias");
    dump(conv3->weight, "conv3.weight");
    dump(conv3->bias, "conv3.bias");
}

// value, policy
Tensor NetImpl::forward(Tensor x, bool print) {
    auto padopts = torch::nn::functional::PadFuncOptions({1, 1, 1, 1});

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv1(x));

    if (print) {
        fmt::print("after conv1:\n{}\n", x);
    }

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv2(x));

    if (print) {
        fmt::print("after conv2:\n{}\n", x);
    }

    x = nn::functional::pad(x, padopts);
    x = torch::relu(conv3(x));

    if (print) {
        fmt::print("after conv3:\n{}\n", x);
    }

    x = flat(x);
    x = torch::log_softmax(x, -1);

    return x;
}

Tensor NetImpl::manual_forward(Tensor x) {
    auto padopts = torch::nn::functional::PadFuncOptions({1, 1, 1, 1});

    x = nn::functional::pad(x, padopts);

    std::vector<Tensor> conv1_outs{};
    for (int i = 0; i < 20; i += 1) {
        Tensor filter = conv1->weight.index({Slice(i, i + 1), "..."});
        Tensor bias = conv1->bias.index({i});
        Tensor conved = nn::functional::conv2d(x, filter);
        Tensor biased = conved + bias;
        conv1_outs.push_back(biased);
    }
    Tensor conv1_out = torch::relu(torch::cat(conv1_outs));

    fmt::print("after mconv1:\n{}\n", conv1_out);
    /*******************************************************/

    // 20,1,8,8
    conv1_out = nn::functional::pad(conv1_out, padopts);

    std::vector<Tensor> conv2_outs{};
    for (int i = 0; i < 20; i += 1) {
        Tensor summed = torch::zeros(
            {1, 1, 6, 6}, torch::TensorOptions().dtype(torch::kFloat32));
        for (int j = 0; j < 20; j += 1) {
            Tensor featmap = conv1_out.index({Slice(j, j + 1), "..."});
            Tensor filter =
                conv2->weight.index({Slice(i, i + 1), Slice(j, j + 1), "..."});
            Tensor conved = nn::functional::conv2d(featmap, filter);
            summed += conved;
        }
        Tensor bias = conv2->bias.index({i});
        Tensor biased = summed + bias;
        conv2_outs.push_back(biased);
    }
    Tensor conv2_out = torch::relu(torch::cat(conv2_outs));

    fmt::print("after mconv2:\n{}\n", conv2_out);
    /*******************************************************/

    conv2_out = nn::functional::pad(conv2_out, padopts);

    Tensor summed = torch::zeros({1, 1, 6, 6},
                                 torch::TensorOptions().dtype(torch::kFloat32));
    for (int i = 0; i < 20; i += 1) {
        Tensor featmap = conv2_out.index({Slice(i, i + 1), "..."});
        Tensor filter = conv3->weight.index({Slice(), Slice(i, i + 1), "..."});
        Tensor conved = nn::functional::conv2d(featmap, filter);
        summed += conved;
    }
    Tensor bias = conv3->bias.index({0});
    Tensor biased = summed + bias;

    Tensor conv3_out = biased;

    fmt::print("after mconv3:\n{}\n", conv3_out);
    /*******************************************************/

    return torch::log_softmax(flat(conv3_out), -1);
}
