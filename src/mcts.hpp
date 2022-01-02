#pragma once

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

enum class Player : uint8_t { White, Black };
enum class Stone : uint8_t { None, White, Black };
bool operator==(Stone stone, Player player);
bool operator==(Player player, Stone stone);
Player operator!(Player player);
inline Stone player_to_stone(Player player) {
    if (player == Player::White) {
        return Stone::White;
    } else {
        return Stone::Black;
    }
}
std::ostream& operator<<(std::ostream& out, Player player);
char stone_char(Stone stone);

struct Action {
    inline Action(int i, int j) : i(i), j(j) {}
    int i, j;
};
std::ostream& operator<<(std::ostream& out, Action action);

struct State {
    std::array<std::array<Stone, 6>, 6> board;
    Player next;
    std::optional<Player> winner;
    int age = 0;

    bool is_ended();
    std::vector<Action> get_actions();
    void place(Action action);
    friend std::ostream& operator<<(std::ostream& out, const State& state);
};
std::ostream& operator<<(std::ostream& out, const State& state);

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
