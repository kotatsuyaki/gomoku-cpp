#include <c10/core/DeviceType.h>
#include <iostream>
#include <torch/torch.h>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    tensor.to(torch::kCUDA);
    std::cout << tensor << std::endl;
}
