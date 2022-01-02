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
    Node(State state, int visits, float value,
         std::optional<Action> last_action);
    State state;
    int visits;
    float value;
    std::optional<Action> last_action;
    std::vector<std::shared_ptr<Node>> children;
};

using NodePtr = std::shared_ptr<Node>;

class Mcts {
    Mcts(Net net);
    Action query(State state);

  private:
    Net net;

    NodePtr select(NodePtr current);
};
