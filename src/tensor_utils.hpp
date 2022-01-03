#pragma once
#include <array>

#include <torch/torch.h>

using Policy = std::array<float, 36>;
using Canonical = std::array<std::array<float, 6>, 6>;

Policy policy_from_tensor(torch::Tensor tensor);
float value_from_tensor(torch::Tensor tensor);
void show_policy(Policy policy);
void show_iters();
