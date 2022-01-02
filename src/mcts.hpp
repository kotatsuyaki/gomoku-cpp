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
};

using NodePtr = std::shared_ptr<Node>;

class Mcts {
  public:
    Mcts(Net net);
    Action query(State state);

  private:
    Net net;

    NodePtr select(NodePtr current);
    void expand(NodePtr current);
    void evaluate(NodePtr current, Player me);
};
