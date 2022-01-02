#include "game.hpp"

#include <cassert>

/* state implementation */
// constructors
State::State() {}
State::State(const State& rhs) {
    board = rhs.board;
    next = rhs.next;
    winner = rhs.winner;
    age = rhs.age;
}
State& State::operator=(const State& rhs) {
    board = rhs.board;
    next = rhs.next;
    winner = rhs.winner;
    age = rhs.age;
    return *this;
}

// getters
bool State::is_ended() const { return (age == 36) || (winner.has_value()); }
int State::get_age() const { return age; }
std::optional<Player> State::get_winner() const { return winner; }
Player State::get_next() const { return next; }
std::array<std::array<float, 6>, 6> State::canonical() const {
    std::array<std::array<float, 6>, 6> arr;
    for (size_t i = 0; i < 6; i += 1) {
        for (size_t j = 0; j < 6; j += 1) {
            if (board[i][j] == next) {
                arr[i][j] = 1.0f;
            } else if (board[i][j] == !next) {
                arr[i][j] = -1.0f;
            } else {
                assert(board[i][j] == Stone::None);
                arr[i][j] = 0.0f;
            }
        }
    }
    return arr;
}

// list out actions
std::vector<Action> State::get_actions() const {
    std::vector<Action> actions{};
    for (size_t i = 0; i < 6; i += 1) {
        for (size_t j = 0; j < 6; j += 1) {
            if (board[i][j] == Stone::None) {
                actions.push_back(Action(i, j));
            }
        }
    }
    return actions;
}

// perform action
void State::place(Action action) {
    assert(action.i < 6 && action.i >= 0);
    assert(action.j < 6 && action.j >= 0);
    int i = action.i;
    int j = action.j;
    Player me = next;
    next = !next;
    age += 1;

    board[i][j] = stone_from_player(me);

    /* Vertical winner check */
    int vertical_count = 0;
    // vertical down
    for (int x = i + 0; x <= i + 5 && x < 6; x += 1) {
        if (board[x][j] == me) {
            vertical_count += 1;
        } else {
            break;
        }
    }
    // vertical up
    for (int x = i - 1; x >= i - 5 && x >= 0; x -= 1) {
        if (board[x][j] == me) {
            vertical_count += 1;
        } else {
            break;
        }
    }
    if (vertical_count == 5) {
        winner = me;
        return;
    }

    /* Horizontal winner check */
    int hori_count = 0;
    // horizontal right
    for (int y = j + 0; y <= j + 5 && y < 6; y += 1) {
        if (board[i][y] == me) {
            hori_count += 1;
        } else {
            break;
        }
    }
    // horizontal left
    for (int y = j - 1; y >= j - 5 && y >= 0; y -= 1) {
        if (board[i][y] == me) {
            hori_count += 1;
        } else {
            break;
        }
    }
    if (hori_count == 5) {
        winner = me;
        return;
    }

    /* Main diagonal winner check */
    int main_count = 0;
    // main diag bot-right
    {
        int x = i;
        int y = j;
        while (x <= i + 5 && x < 6 && y < 6) {
            if (board[x][y] == me) {
                main_count += 1;
                x += 1;
                y += 1;
            } else {
                break;
            }
        }
    }
    // main diag top-left
    {
        int x = i - 1;
        int y = j - 1;
        while (x >= i - 5 && x >= 0 && y >= 0) {
            if (board[x][y] == me) {
                main_count += 1;
                x -= 1;
                y -= 1;
            } else {
                break;
            }
        }
    }
    if (main_count == 5) {
        winner = me;
        return;
    }

    /* Second diagonal winner check */
    int sec_count = 0;
    // second diag bot-left (x goes up, y goes down)
    {
        int x = i;
        int y = j;
        while (x <= i + 5 && x < 6 && y >= 0) {
            if (board[x][y] == me) {
                sec_count += 1;
                x += 1;
                y -= 1;
            } else {
                break;
            }
        }
    }
    // second diag top-right (x goes down, y goes up)
    {
        int x = i - 1;
        int y = j + 1;
        while (x >= i - 5 && x >= 0 && y < 6) {
            if (board[x][y] == me) {
                sec_count += 1;
                x -= 1;
                y += 1;
            } else {
                break;
            }
        }
    }
    if (sec_count == 5) {
        winner = me;
        return;
    }
}

/* equality between stone and player */
bool operator==(Stone stone, Player player) {
    bool white_match = (stone == Stone::White && player == Player::White);
    bool black_match = (stone == Stone::Black && player == Player::Black);
    return white_match || black_match;
}
bool operator==(Player player, Stone stone) { return stone == player; }

/* negation of player */
Player operator!(Player player) {
    return (player == Player::White) ? Player::Black : Player::White;
}

/* conversion between stone & player */
Stone stone_from_player(Player player) {
    if (player == Player::White) {
        return Stone::White;
    } else {
        return Stone::Black;
    }
}

/* Print implementations */
std::ostream& operator<<(std::ostream& out, Player player) {
    out << ((player == Player::White) ? "White" : "Black");
    return out;
}

std::ostream& operator<<(std::ostream& out, Stone stone) {
    if (stone == Stone::None) {
        out << ' ';
    } else if (stone == Stone::White) {
        out << 'o';
    } else if (stone == Stone::Black) {
        out << '#';
    } else {
        assert(false);
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, const Action& action) {
    out << "(" << action.i << ", " << action.j << ")";
    return out;
}

std::ostream& operator<<(std::ostream& out, const State& state) {
    for (int i = 0; i < 6; i += 1) {
        for (int j = 0; j < 6; j += 1) {
            out << state.board[i][j];
        }
        out << '\n';
    }
    return out;
}
