#include "mcts.hpp"
#include "model.hpp"

#include <cassert>
#include <cstdlib>
#include <random>
#include <string>

#include <torch/torch.h>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

const auto FGRED = fmt::fg(fmt::color::red);

void randgame();
void combatgame();
void create_net();

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print(stderr, FGRED, "No subcommand supplied\n");
        return EXIT_FAILURE;
    }

    std::string subcmd{argv[1]};
    if (subcmd == "create") {
        create_net();
        return EXIT_SUCCESS;
    } else if (subcmd == "randgame") {
        randgame();
    } else if (subcmd == "combatgame") {
        combatgame();
    } else {
        fmt::print(stderr, FGRED, "unknown subcommand {}\n", subcmd);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void create_net() {
    torch::manual_seed(42);

    auto net = Net();
    auto x = torch::eye(6).reshape({1, 1, 6, 6}).repeat({2, 1, 1, 1});
    auto [value, policy] = net->forward(x);

    fmt::print("input shape = {}\n"
               "value shape = {}\n"
               "policy shape = {}",
               x.sizes(), value.sizes(), policy.sizes());

    torch::save(net, "net.pt");

    {}
}

Action randmove(State state) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    auto actions = state.get_actions();
    std::uniform_int_distribution<> dist(0, actions.size() - 1);
    Action action = actions[dist(gen)];
    return action;
}

void randgame() {
    State state{};
    while (state.is_ended() == false) {
        auto me = state.get_next();
        auto action = randmove(state);
        state.place(action);

        fmt::print("{} placed stone at {}:\n{}\n", me, action, state);
    }
    if (state.get_winner().has_value()) {
        fmt::print("Winner: {}\n", state.get_winner().value());
    } else {
        fmt::print("Tie\n");
    }
}

void combatgame() {
    State state{};
    Net net{};
    torch::load(net, "net.pt");
    Mcts mcts{net};

    while (state.is_ended() == false) {
        auto me = state.get_next();
        if (state.get_age() % 2 == 0) {
            Action action = randmove(state);
            state.place(action);
            fmt::print("{} placed stone at {}:\n{}\n", me, action, state);
        } else {
            Action action = mcts.query(state);
            state.place(action);
            fmt::print("{} placed stone at {}:\n{}\n", me, action, state);
        }
    }
    if (state.get_winner().has_value()) {
        fmt::print("Winner: {}\n", state.get_winner().value());
    } else {
        fmt::print("Tie\n");
    }
}
