#pragma once

#include "game.hpp"
#include "model.hpp"

#include <array>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <vector>

#include <torch/torch.h>

using Policy = std::array<float, 36>;
using Canonical = std::array<std::array<float, 6>, 6>;

struct Node {
    Node(State state, std::optional<Action> last_action,
         std::shared_ptr<Node> parent);
    State state;
    int visits = 0;
    float ttlvalue = 0.0f;

    std::optional<std::array<float, 36>> policy = std::nullopt;
    std::optional<float> value = std::nullopt;

    std::optional<Action> last_action;
    std::shared_ptr<Node> parent;
    std::vector<std::shared_ptr<Node>> children{};
    friend std::ostream& operator<<(std::ostream& out, const Node& node);
};
std::ostream& operator<<(std::ostream& out, const Node& node);

using NodePtr = std::shared_ptr<Node>;

class Mcts {
  public:
    Mcts(Net net);
    std::pair<Action, std::array<float, 36>> query(State state);
    std::pair<Action, std::array<float, 36>> raw_query(State state);

  private:
    Net net;

    NodePtr sample_select(NodePtr current);
    NodePtr max_select(NodePtr current);
    std::vector<float> children_scores(NodePtr current);
    void expand(NodePtr current);
    void evaluate(NodePtr current, Player me);
};

Policy policy_from_tensor(Tensor tensor);
float value_from_tensor(Tensor tensor);
void show_policy(Policy policy);
void show_iters();
