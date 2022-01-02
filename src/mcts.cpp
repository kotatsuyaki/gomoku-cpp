#include "mcts.hpp"

#include <cmath>
#include <limits>

#include <fmt/core.h>

bool operator==(Stone stone, Player player) {
    bool white_match = (stone == Stone::White && player == Player::White);
    bool black_match = (stone == Stone::Black && player == Player::Black);
    return white_match || black_match;
}
bool operator==(Player player, Stone stone) { return stone == player; }
Player operator!(Player player) {
    return (player == Player::White) ? Player::Black : Player::White;
}

Node::Node(State state, int32_t visits, float value,
           std::optional<Action> last_action)
    : state(state), visits(visits), value(value), last_action(last_action),
      children(std::vector<NodePtr>()) {}

Mcts::Mcts(Net net) : net(net) {}

Action Mcts::query(State state) {
    NodePtr root = std::make_shared<Node>(state, 0, 0.0f, std::nullopt);
    Player me = state.next;

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

std::vector<Action> State::get_actions() {
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

bool State::is_ended() { return (age == 36) || (winner.has_value()); }

void State::place(Action action) {
    assert(action.i < 6 && action.i >= 0);
    assert(action.j < 6 && action.j >= 0);
    int i = action.i;
    int j = action.j;
    Player me = next;
    next = !next;
    age += 1;

    board[i][j] = player_to_stone(me);

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
        fmt::print("vertical win\n");
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
        fmt::print("horizontal win\n");
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
        fmt::print("main diag win\n");
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
        fmt::print("second diag win\n");
        return;
    }
}

std::ostream& operator<<(std::ostream& out, Player player) {
    out << ((player == Player::White) ? "White" : "Black");
    return out;
}
std::ostream& operator<<(std::ostream& out, Action action) {
    out << "(" << action.i << ", " << action.j << ")";
    return out;
}

std::ostream& operator<<(std::ostream& out, const State& state) {
    for (int i = 0; i < 6; i += 1) {
        for (int j = 0; j < 6; j += 1) {
            out << stone_char(state.board[i][j]);
        }
        out << '\n';
    }
    return out;
}

char stone_char(Stone stone) {
    if (stone == Stone::None) {
        return ' ';
    } else if (stone == Stone::White) {
        return 'o';
    } else if (stone == Stone::Black) {
        return '#';
    }
    assert(false);
}
