#include "tensor_utils.hpp"

#include <fmt/core.h>

Policy policy_from_tensor(torch::Tensor tensor) {
    Policy policy;
    std::memcpy(policy.data(),
                static_cast<float*>(tensor.to(torch::kCPU).data_ptr()),
                sizeof(policy));
    return policy;
}

float value_from_tensor(torch::Tensor tensor) {
    float value = *static_cast<float*>(tensor.to(torch::kCPU).data_ptr());
    return value;
}

void show_policy(Policy policy) {
    for (int i = 0; i < 36; i += 1) {
        fmt::print("{:8.2}", policy[i]);
        if ((i + 1) % 6 == 0) {
            fmt::print("\n");
        } else {
            fmt::print(", ");
        }
    }
}
