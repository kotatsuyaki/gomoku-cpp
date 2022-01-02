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

static const char* ITERS_S = std::getenv("ITERS");
static int ITERS = ITERS_S ? std::atoi(ITERS_S) : 500;
static const char* RAWQUERY_S = std::getenv("RAWQUERY");
static bool RAWQUERY = RAWQUERY_S ? true : false;

std::pair<Action, std::array<float, 36>> Mcts::query(State state) {
    if (RAWQUERY) {
        return raw_query(state);
    }

    NodePtr root = std::make_shared<Node>(state, std::nullopt);
    Player me = state.get_next();

    for (int iter = 0; iter < 1000; iter += 1) {
        NodePtr current = root;

        // select
        while (current->value.has_value()) {
            NodePtr selected = max_select(current);
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
                score_update =
                    (winner.value() == current->state.get_next()) ? 1.0f : 0.0f;
            } else {
                score_update = 0.5f;
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
        int i = child->last_action.value().i;
        int j = child->last_action.value().j;

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

    auto max_child = sample_select(root);
    return {max_child->last_action.value(), policy};
}

std::pair<Action, std::array<float, 36>> Mcts::raw_query(State state) {
    auto canonical = state.canonical();
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto input = torch::from_blob(canonical.data(), {1, 1, 6, 6}, options)
                     .to(torch::kCUDA);

    auto [value_t, policy_t] = net->forward(input);
    policy_t = policy_t.exp().to(torch::kCPU);
    Policy policy = policy_from_tensor(policy_t);

    auto actions = state.get_actions();
    auto action = std::max_element(
        actions.begin(), actions.end(), [&](Action a, Action b) {
            return policy[a.i * 6 + a.j] < policy[b.i * 6 + b.j];
        });

    return {*action, policy};
}

std::vector<float> Mcts::children_scores(NodePtr current) {
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

    return scores;
}

NodePtr Mcts::sample_select(NodePtr current) {
    auto scores = children_scores(current);

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

NodePtr Mcts::max_select(NodePtr current) {
    auto scores = children_scores(current);
    int idx = std::max_element(scores.begin(), scores.end()) - scores.begin();

    return current->children[idx];
}

void Mcts::expand(NodePtr current) {
    auto actions = current->state.get_actions();

    for (auto action : actions) {
        State to_state = current->state;
        to_state.place(action);

        auto child = std::make_shared<Node>(to_state, action, current);
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
    Policy policy = policy_from_tensor(policy_t);

    current->value = value;
    current->policy = policy;
}

Policy policy_from_tensor(Tensor tensor) {
    Policy policy;
    std::memcpy(policy.data(),
                static_cast<float*>(tensor.to(torch::kCPU).data_ptr()),
                sizeof(policy));
    return policy;
}

float value_from_tensor(Tensor tensor) {
    float value = *static_cast<float*>(tensor.to(torch::kCPU).data_ptr());
    return value;
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

void show_iters() { fmt::print("Using ITERS = {}\n", ITERS); }

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
