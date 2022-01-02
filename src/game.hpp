#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <ostream>
#include <vector>

enum class Player : uint8_t { White, Black };
enum class Stone : uint8_t { None, White, Black };

// equality between player & stone
bool operator==(Stone stone, Player player);
bool operator==(Player player, Stone stone);

// opponent of player
Player operator!(Player player);

// player-stone conversion
Stone stone_from_player(Player player);

// display for player & stone
std::ostream& operator<<(std::ostream& out, Player player);
std::ostream& operator<<(std::ostream& out, Stone stone);

class Action {
  public:
    inline Action(int i, int j) : i(i), j(j) {}
    int i, j;

    friend std::ostream& operator<<(std::ostream& out, const Action& action);
};
std::ostream& operator<<(std::ostream& out, const Action& action);

class State {
  private:
    std::array<std::array<Stone, 6>, 6> board{};
    Player next = Player::White;
    std::optional<Player> winner = std::nullopt;
    int age = 0;

  public:
    State();
    State(const State& rhs);
    State& operator=(const State& rhs);

    bool is_ended() const;
    std::vector<Action> get_actions() const;
    void place(Action action);

    int get_age() const;
    Player get_next() const;
    std::optional<Player> get_winner() const;
    std::array<std::array<float, 6>, 6> canonical() const;

    friend std::ostream& operator<<(std::ostream& out, const State& state);
};
std::ostream& operator<<(std::ostream& out, const State& state);
