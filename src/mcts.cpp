#include "mcts.hpp"
#include "tensor_utils.hpp"

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
    : state(state), last_action(last_action), parent(parent),
      depth((parent != nullptr) ? (parent->depth + 1) : 0) {}

Mcts::Mcts() {}

static const char* ITERS_S = std::getenv("ITERS");
static int ITERS = ITERS_S ? std::atoi(ITERS_S) : 5000;

std::pair<Action, std::array<float, 36>> Mcts::query(State state) {
    NodePtr root = std::make_shared<Node>(state, std::nullopt);
    Player me = state.get_next();

    for (int iter = 0; iter < ITERS; iter += 1) {
        NodePtr current = root;

        // select
        while (current->children.empty() == false) {
            NodePtr selected = max_select(current);
            current = selected;
        }

        // expand
        if (current->state.is_ended() == false) {
            expand(current);
        }

        // simulate
        float score_update;
        auto [depth, winner] = simulate(current);

        // backprop
        while (true) {
            score_update = 1.0f - score_update;

            current->visits += 1;
            if (winner.has_value()) {
                if (current->state.get_next() == winner.value()) {
                    // this leads to a win => the parent node don't want this
                    current->ttlvalue +=
                        -std::exp(-0.5f * (0.2f * depth - 5.0f)) - 5.0f;
                } else {
                    // this leads to a lose => the parent node is happy
                    current->ttlvalue += 1.0f;
                }
            } else {
                current->ttlvalue += 0.2;
            }

            if (current->parent.lock() != nullptr) {
                current = current->parent.lock();
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
    fmt::print("Raw policy = {}\n", fmt::join(policy, ", "));

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

std::vector<float> Mcts::children_scores(NodePtr current) {
    std::vector<float> scores(current->children.size(), 0.0f);
    float parent_visits = static_cast<float>(current->visits);

    for (size_t i = 0; i < current->children.size(); i += 1) {
        NodePtr child = current->children[i];
        assert(child->last_action.has_value());
        Action act = child->last_action.value();

        float child_visits = static_cast<float>(child->visits);

        float exploit = child->ttlvalue / (child_visits + 1.0f);
        float explore =
            std::sqrt(2.0f * std::log(std::max(1.0f, parent_visits)) /
                      (child_visits + 1.0f));

        float ucb = exploit + explore;
        scores[i] = ucb;
    }

    return scores;
}

NodePtr Mcts::sample_select(NodePtr current) {
    auto scores = children_scores(current);
    int idx = std::max_element(scores.begin(), scores.end()) - scores.begin();

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

std::pair<int, std::optional<Player>> Mcts::simulate(NodePtr current) {
    // state is to be modified in-place
    State state{current->state};
    int i = 0;
    while (state.is_ended() == false) {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        i += 1;

        auto actions = state.get_actions();
        std::uniform_int_distribution<> dist(0, actions.size() - 1);
        Action action = actions[dist(gen)];
        state.place(action);
    }
    return {current->depth + i, state.get_winner()};
}

std::ostream& operator<<(std::ostream& out, const Node& node) {
    out << "Node("
        << "visits = " << node.visits << ", ttlvalue = " << node.ttlvalue
        << ")";
    return out;
}

void show_iters() { fmt::print("Using ITERS = {}\n", ITERS); }
