#include "mcts.hpp"

#include <cmath>
#include <limits>

#include <fmt/core.h>

Node::Node(State state, int32_t visits, float value,
           std::optional<Action> last_action)
    : state(state), visits(visits), value(value), last_action(last_action),
      children(std::vector<NodePtr>()) {}

Mcts::Mcts(Net net) : net(net) {}

Action Mcts::query(State state) {
    NodePtr root = std::make_shared<Node>(state, 0, 0.0f, std::nullopt);
    Player me = state.get_next();

    for (int iter = 0; iter < 1000; iter += 1) {
        NodePtr current = root;

        // select
        while (current->children.empty() == false) {
            NodePtr selected = select(current);
            current = selected;
        }

        // expand
        // if
    }
}

NodePtr Mcts::select(NodePtr current) {
    std::vector<float> scores(current->children.size(), 0.0f);
    float parent_visits = static_cast<float>(current->visits);

    for (size_t i = 0; i < current->children.size(); i += 1) {
        NodePtr child = current->children[i];
        float child_visits = static_cast<float>(child->visits);

        float exploit = child->value / (child_visits + 1.0f);
        float explore =
            std::sqrt(2.0 * std::log(std::max(parent_visits, 1.0f)) /
                      (child_visits + 1.0f));
        float ucb = exploit + explore;
        scores[i] = ucb;
    }

    std::optional<NodePtr> selected;
    float max_score = -std::numeric_limits<float>::max();

    for (size_t i = 0; i < current->children.size(); i += 1) {
        float score = scores[i];
        if (score >= max_score) {
            max_score = score;
            selected = current->children[i];
        }
    }

    return selected.value();
}
