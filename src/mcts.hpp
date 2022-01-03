#pragma once

#include "game.hpp"
#include "model.hpp"
#include "tensor_utils.hpp"

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
    int depth = 0;

    std::optional<Action> last_action;
    std::weak_ptr<Node> parent;
    std::vector<std::shared_ptr<Node>> children{};

    friend std::ostream& operator<<(std::ostream& out, const Node& node);
};
std::ostream& operator<<(std::ostream& out, const Node& node);

using NodePtr = std::shared_ptr<Node>;

class Mcts {
  public:
    Mcts();
    std::pair<Action, std::array<float, 36>> query(State state);

  private:
    NodePtr sample_select(NodePtr current);
    NodePtr max_select(NodePtr current);
    std::vector<float> children_scores(NodePtr current);
    void expand(NodePtr current);
    // depth and winner
    std::pair<int, std::optional<Player>> simulate(NodePtr current);
};

void show_iters();
