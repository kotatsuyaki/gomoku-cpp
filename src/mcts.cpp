#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <random>

#include <numeric>
#include <torch/torch.h>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

const auto FGRED = fmt::fg(fmt::color::red);

Node::Node(State state, std::optional<Action> last_action,
           std::shared_ptr<Node> parent = nullptr)
    : state(state), last_action(last_action), parent(parent) {}

Mcts::Mcts(Net net) : net(net) {}

std::pair<Action, std::array<float, 36>> Mcts::query(State state) {
    NodePtr root = std::make_shared<Node>(state, std::nullopt);
    Player me = state.get_next();

    for (int iter = 0; iter < 1000; iter += 1) {
        NodePtr current = root;

        // select
        while (current->value.has_value()) {
            NodePtr selected = select(current);
            current = selected;
        }

        // expand and get amount of score update
        float score_update;
        if (current->state.is_ended() == false) {
            expand(current);
            evaluate(current, me);

            assert(current->value.has_value());
            score_update = current->value.value();
        } else {
            auto winner = current->state.get_winner();
            if (winner.has_value()) {
                score_update = (winner.value() == current->state.get_next())
                                   ? 1.0f
                                   : -1.0f;
            } else {
                score_update = 0.0f;
            }
        }

        // backprop
        while (true) {
            score_update = 1.0 - score_update;

            current->visits += 1;
            current->ttlvalue += score_update;
            if (current->parent != nullptr) {
                current = current->parent;
            } else {
                break;
            }
        }
    } // end loop

    // calculuate policy
    std::array<float, 36> policy{};
    for (auto child : root->children) {
        assert(child->last_action.has_value());
        int i = child->last_action->i;
        int j = child->last_action->j;

        float visits = static_cast<float>(child->visits);
        policy[i * 6 + j] = visits;
    }
    // make it sum up to 1
    float sum = std::accumulate(policy.begin(), policy.end(), 0.0f);
    if (sum >= 1.0f) {
        for (float& item : policy) {
            float after = item / sum;
            if (item != item) {
                fmt::print(FGRED, "Got nan: {} / {} = {}\n", item, sum, after);
            }
            item = after;
        }
    }

#ifdef FINDMAX
    // find max child
    std::vector<float> scores(root->children.size(), 0.0f);
    auto max_child = *std::max_element(
        root->children.begin(), root->children.end(),
        [](NodePtr& a, NodePtr& b) -> bool { return a->visits < b->visits; });
    assert(max_child->last_action.has_value());
#else
    auto max_child = select(root);
#endif

    return {max_child->last_action.value(), policy};
}

NodePtr Mcts::select(NodePtr current) {
    assert(current->policy.has_value());

    std::vector<float> scores(current->children.size(), 0.0f);
    float parent_visits = static_cast<float>(current->visits);
    auto parent_policy = current->policy.value();

    for (size_t i = 0; i < current->children.size(); i += 1) {
        NodePtr child = current->children[i];
        assert(child->last_action.has_value());
        Action act = child->last_action.value();

        float child_visits = static_cast<float>(child->visits);

        float exploit = child->ttlvalue / (child_visits + 1.0f);
        float explore = std::sqrt(2.0) * parent_policy[act.i * 6 + act.j] *
                        std::sqrt(parent_visits) / (1.0f + child_visits);

        float ucb = exploit + explore;
        scores[i] = ucb;
    }

    /* sample an index */
    // total sum
    float sum = std::accumulate(scores.begin(), scores.end(), 0.0f);
    // prefix sum
    std::vector<float> scores_prefix_sum{};
    std::exclusive_scan(scores.begin(), scores.end(),
                        std::back_inserter(scores_prefix_sum), 0.0f);

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0f, sum);
    auto it = std::upper_bound(scores_prefix_sum.begin(),
                               scores_prefix_sum.end(), dist(gen));
    int idx = it - scores_prefix_sum.begin() - 1;

    // not sure if it generates OOB access
    assert(idx < 0 || idx >= current->children.size());

    return current->children[idx];
}

void Mcts::expand(NodePtr current) {
    auto actions = current->state.get_actions();

    for (auto action : actions) {
        State to_state = current->state;
        to_state.place(action);

        auto child = std::make_shared<Node>(to_state, action);
        current->children.push_back(child);
    }
}

void Mcts::evaluate(NodePtr current, Player me) {
    auto canonical = current->state.canonical();
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto input = torch::from_blob(canonical.data(), {1, 1, 6, 6}, options)
                     .to(torch::kCUDA);

    auto [value_t, policy_t] = net->forward(input);
    policy_t = policy_t.exp().to(torch::kCPU);
    value_t = value_t.to(torch::kCPU);

    float value = *static_cast<float*>(value_t.data_ptr());
    std::array<float, 36> policy;
    std::memcpy(policy.data(), policy_t.data_ptr(), sizeof(policy));

    current->value = value;
    current->policy = policy;
}

std::ostream& operator<<(std::ostream& out, const Node& node) {
    out << "Node("
        << "visits = " << node.visits << ", ttlvalue = " << node.ttlvalue
        << ", value = "
        << (node.value.has_value() ? fmt::format("{:.3}", node.value.value())
                                   : "none")
        << ", policy = "
        << (node.policy.has_value()
                ? fmt::format("{}", fmt::join(node.policy.value(), ", "))
                : "none")
        << ")";
    return out;
}
