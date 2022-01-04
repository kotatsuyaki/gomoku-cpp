#include "tensor_utils.hpp"

#include <algorithm>
#include <fmt/core.h>
#include <fmt/format.h>

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

void show_canonical(Canonical canonical) {
    for (int i = 0; i < 6; i += 1) {
        fmt::print("{:8.2}\n", fmt::join(canonical[i], ", "));
    }
}

std::vector<std::pair<Canonical, Policy>> augment(Canonical state,
                                                  Policy policy) {
    std::vector<std::pair<Canonical, Policy>> pairs;
    pairs.push_back({state, policy});

    for (int rot = 1; rot < 3; rot += 1) {
        auto& [last_state, last_policy] = pairs.back();
        Canonical augmented_state{};
        Policy augmented_policy{};

        // counter 90
        for (int i = 0; i < 6; i += 1) {
            for (int j = 0; j < 6; j += 1) {
                augmented_state[i][j] = last_state[j][6 - 1 - i];
                augmented_policy[i * 6 + j] =
                    last_policy[(j * 6) + (6 - 1 - i)];
            }
        }

        pairs.push_back({augmented_state, augmented_policy});
    }

    return pairs;
}
