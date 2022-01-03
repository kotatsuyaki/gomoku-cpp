#include "net_query.hpp"

NetQuery::NetQuery(Net net) : net(net){}

std::pair<Action, Policy> NetQuery::raw_query(State state) {
    auto canonical = state.canonical();
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto input = torch::from_blob(canonical.data(), {1, 1, 6, 6}, options)
                     .to(torch::kCUDA);

    auto policy_t = net->forward(input);
    policy_t = policy_t.exp().to(torch::kCPU);
    Policy policy = policy_from_tensor(policy_t);

    auto actions = state.get_actions();
    auto action = std::max_element(
        actions.begin(), actions.end(), [&](Action a, Action b) {
            return policy[a.i * 6 + a.j] < policy[b.i * 6 + b.j];
        });

    return {*action, policy};
}
