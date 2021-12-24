#include <cstdlib>
#include <iostream>

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <fmt/core.h>
#include <fmt/ostream.h>

int main(int argc, char** argv) {
    torch::Tensor tensor = torch::rand({2, 3});
    tensor = tensor.to(torch::kCUDA);
    fmt::print("{}", tensor);

    return EXIT_SUCCESS;
}
