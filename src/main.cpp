#include "mcts.hpp"
#include "model.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <random>
#include <string>

#include <torch/torch.h>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

const auto FGRED = fmt::fg(fmt::color::red);
const auto FGGRN = fmt::fg(fmt::color::green);

void randgame();
void combatgame();
void create();
void humangame();
void train();
void train2();
void bench();
void dump();

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print(stderr, FGRED, "No subcommand supplied\n");
        return EXIT_FAILURE;
    }

    std::string subcmd{argv[1]};
    if (subcmd == "create") {
        create();
        return EXIT_SUCCESS;
    } else if (subcmd == "randgame") {
        randgame();
    } else if (subcmd == "combatgame") {
        combatgame();
    } else if (subcmd == "train") {
        train();
    } else if (subcmd == "bench") {
        bench();
    } else if (subcmd == "humangame") {
        humangame();
    } else if (subcmd == "dump") {
        dump();
    } else {
        fmt::print(stderr, FGRED, "unknown subcommand {}\n", subcmd);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void create() {
    if (const char* seed = std::getenv("SEED")) {
        fmt::print("Using supplied seed {}\n", seed);
        torch::manual_seed(std::atoi(seed));
    } else {
        fmt::print("Using default seed\n");
        torch::manual_seed(42);
    }

    fmt::print("Creating net\n");

    auto net = Net();
    auto x = torch::eye(6).reshape({1, 1, 6, 6}).repeat({2, 1, 1, 1});
    auto policy = net->forward(x);

    fmt::print("input shape = {}\n"
               "policy shape = {}\n",
               x.sizes(), policy.sizes());

    torch::save(net, "net.pt");

    fmt::print("Creating optimizer\n");
    auto adam_opts = torch::optim::AdamOptions(1e-4);
    torch::optim::Adam opt(net->parameters(), adam_opts);
    torch::save(opt, "opt.pt");
}

Action randmove(State state) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    auto actions = state.get_actions();
    std::uniform_int_distribution<> dist(0, actions.size() - 1);
    Action action = actions[dist(gen)];
    return action;
}

void show_winner(State state) {
    if (state.get_winner().has_value()) {
        fmt::print("\rWinner: {}\n", state.get_winner().value());
    } else {
        fmt::print("\rGame tie\n");
    }
}

void randgame() {
    State state{};
    while (state.is_ended() == false) {
        auto me = state.get_next();
        auto action = randmove(state);
        state.place(action);

        fmt::print("{} placed stone at {}:\n{}\n", me, action, state);
    }
    show_winner(state);
}

void humangame() {
    State state{};
    Mcts mcts{};

    while (state.is_ended() == false) {
        auto me = state.get_next();
        if (state.get_age() % 2 == 0) {
            auto [action, policy] = mcts.query(state);
            state.place(action);

            fmt::print("Policy = {}\n", fmt::join(policy, ", "));
            fmt::print("{} placed stone at {}:\n{}\n", me, action, state);
        } else {
            int i, j;
            fmt::print("input i and j:\n");
            std::cin >> i >> j;
            Action action(i, j);
            state.place(action);

            fmt::print("{} placed stone at {}:\n{}\n", me, action, state);
        }
    }
    show_winner(state);
}

void combatgame() {
    State state{};
    Net net{};
    torch::load(net, "net.pt");
    Mcts mcts{};
    net->to(torch::kCUDA);

    while (state.is_ended() == false) {
        auto me = state.get_next();
        if (state.get_age() % 2 == 0) {
            Action action = randmove(state);
            state.place(action);
            fmt::print("{} placed stone at {}:\n{}\n", me, action, state);
        } else {
            Action action = mcts.query(state).first;
            state.place(action);
            fmt::print("{} placed stone at {}:\n{}\n", me, action, state);
        }
    }
    show_winner(state);
}

void train() {
    // print setup info
    int plays;
    if (const char* plays_s = getenv("PLAYS")) {
        fmt::print("Using supplied plays {}\n", plays_s);
        plays = std::atoi(plays_s);
    } else {
        fmt::print("Using default plays 8\n");
        plays = 8;
    }
    show_iters();

    // load net
    Net net{};
    fmt::print("Loading model and optimizer\n");
    torch::load(net, "net.pt");
    torch::optim::Adam opt(net->parameters());
    torch::load(opt, "opt.pt");
    net->to(torch::kCUDA);

    Mcts mcts{};
    std::vector<std::pair<Canonical, Policy>> s_p_pairs{};

#pragma omp parallel for
    for (int i = 0; i < plays; i += 1) {
        fmt::print("Selfplay game #{} started\n", i);
        std::vector<std::pair<State, Policy>> local_s_p_pairs{};

        State state{};
        while (state.is_ended() == false) {
            std::cout.flush();

            auto me = state.get_next();
            auto [action, policy] = mcts.query(state);

            local_s_p_pairs.emplace_back(state, policy);

            state.place(action);
        }

        fmt::print("Selfplay game #{} ended\n", i);
        show_winner(state);

#pragma omp critical
        {
            for (auto item : s_p_pairs) {
                s_p_pairs.push_back(item);
            }
        }
    }

    // Shuffle training data
    fmt::print("Shuffling {} history samples\n", s_p_pairs.size());
    std::random_shuffle(s_p_pairs.begin(), s_p_pairs.end());

    // Train
    fmt::print("Training on {} history samples\n", s_p_pairs.size());
    int i;
    for (auto [s, p] : s_p_pairs) {
        net->zero_grad();
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto state_t =
            torch::from_blob(s.data(), {1, 1, 6, 6}, options).to(torch::kCUDA);
        auto policy_t =
            torch::from_blob(p.data(), {1, 36}, options).to(torch::kCUDA);

        // fmt::print("s/v/p = {}, {}, {}\n", state_t, value_t, policy_t);
        auto policy_p = net->forward(state_t);
        // fmt::print("vp/pp = {}, {}\n", value_p, policy_p);

        auto ploss = -(policy_p * policy_t).sum(torch::kFloat32);
        auto loss = ploss;
        if (*static_cast<bool*>(
                (loss != loss).any().to(torch::kCPU).data_ptr())) {
            fmt::print(FGRED, "Got nan in loss (shape = {}): {}\n",
                       loss.sizes(), loss);
            assert(false);
        }

        // calculate gradients
        loss.backward();
        // update params
        opt.step();

        i += 1;
        if (i % 10 == 0) {
            fmt::print(FGGRN, "Trained {} iterations\n", i);
            float ploss_s =
                *static_cast<float*>(ploss.to(torch::kCPU).data_ptr());
            float loss_s =
                *static_cast<float*>(loss.to(torch::kCPU).data_ptr());
            fmt::print("Loss {:.3} = {:.3}\n", loss_s, ploss_s);
        }
    }

    fmt::print("Saving model and optimizer\n");
    torch::save(net, "net.pt");
    torch::save(opt, "opt.pt");
}

void train2() {
    // print setup info
    int plays;
    if (const char* plays_s = getenv("PLAYS")) {
        fmt::print("Using supplied plays {}\n", plays_s);
        plays = std::atoi(plays_s);
    } else {
        fmt::print("Using default plays 8\n");
        plays = 8;
    }
    show_iters();

    // load net
    Net net{};
    fmt::print("Loading model and optimizer\n");
    torch::load(net, "net.pt");
    torch::optim::Adam opt(net->parameters());
    torch::load(opt, "opt.pt");
    net->to(torch::kCUDA);

    Mcts mcts{};

    // collect self-play data
    std::vector<std::tuple<Canonical, float, Policy>> s_v_p_tuples{};
    for (int i = 0; i < plays; i += 1) {
        fmt::print("Selfplay game #{}\n", i);

        std::vector<std::pair<State, Policy>> s_p_pairs{};

        State state{};
        while (state.is_ended() == false) {
            fmt::print("\rAge: {}", state.get_age());
            std::cout.flush();

            auto me = state.get_next();
            auto [action, policy] = mcts.query(state);

            s_p_pairs.emplace_back(state, policy);

            state.place(action);
        }

        show_winner(state);
        auto winner = state.get_winner();
        for (auto [state, policy] : s_p_pairs) {
            Player me = state.get_next();
            float value;
            if (winner.has_value()) {
                value = (winner.value() == me) ? 1.0f : 0.0f;
            } else {
                value = 0.5f;
            }
            s_v_p_tuples.emplace_back(state.canonical(), value, policy);
        }
    }

    fmt::print("Shuffling {} history samples\n", s_v_p_tuples.size());
    std::random_shuffle(s_v_p_tuples.begin(), s_v_p_tuples.end());
    int i = 0;

    fmt::print("Training on {} history samples\n", s_v_p_tuples.size());
    for (auto [s, v, p] : s_v_p_tuples) {
        net->zero_grad();
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto state_t =
            torch::from_blob(s.data(), {1, 1, 6, 6}, options).to(torch::kCUDA);
        auto value_t = torch::from_blob(&v, {1, 1}, options).to(torch::kCUDA);
        auto policy_t =
            torch::from_blob(p.data(), {1, 36}, options).to(torch::kCUDA);

        // fmt::print("s/v/p = {}, {}, {}\n", state_t, value_t, policy_t);
        auto policy_p = net->forward(state_t);
        // fmt::print("vp/pp = {}, {}\n", value_p, policy_p);

        auto ploss = -(policy_p * policy_t).sum(torch::kFloat32);
        auto loss = ploss;
        if (*static_cast<bool*>(
                (loss != loss).any().to(torch::kCPU).data_ptr())) {
            fmt::print(FGRED, "Got nan in loss (shape = {}): {}\n",
                       loss.sizes(), loss);
            assert(false);
        }

        // calculate gradients
        loss.backward();
        // update params
        opt.step();

        i += 1;
        if (i % 10 == 0) {
            fmt::print(FGGRN, "Trained {} iterations\n", i);
            float ploss_s =
                *static_cast<float*>(ploss.to(torch::kCPU).data_ptr());
            float loss_s =
                *static_cast<float*>(loss.to(torch::kCPU).data_ptr());
            fmt::print("Loss {:.3} = {:.3}\n", loss_s, ploss_s);
        }
    }

    fmt::print("Saving model and optimizer\n");
    torch::save(net, "net.pt");
    torch::save(opt, "opt.pt");
}

void bench() {
    // load net
    Net net{};
    fmt::print("Loading model and optimizer\n");
    torch::load(net, "net.pt");
    torch::optim::Adam opt(net->parameters());
    torch::load(opt, "opt.pt");
    Mcts mcts{};
    net->to(torch::kCPU);

    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        Policy board{
            0, 0, 0, 0, 0,  0, //
            1, 1, 1, 1, 0,  0, //
            0, 0, 0, 0, -1, 0, //
            0, 0, 0, 0, -1, 0, //
            0, 0, 0, 0, -1, 0, //
            0, 0, 0, 0, -1, 0,
        };
        auto policy =
            net->forward(torch::from_blob(board.data(), {1, 1, 6, 6}, options));

        fmt::print("Situation:\n");
        show_policy(board);

        fmt::print("policy:\n");
        show_policy(policy_from_tensor(policy.exp()));
    }

    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        Policy board{
            0,  0,  1,  0,  0, 0, //
            -1, -1, -1, -1, 0, 0, //
            0,  0,  0,  0,  0, 0, //
            0,  0,  0,  0,  0, 0, //
            0,  0,  1,  0,  0, 0, //
            0,  0,  0,  1,  0, 0,
        };
        auto policy =
            net->forward(torch::from_blob(board.data(), {1, 1, 6, 6}, options));

        fmt::print("Situation:\n");
        show_policy(board);

        fmt::print("policy:\n");
        show_policy(policy_from_tensor(policy.exp()));
    }

    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        Policy board{
            0, 0,  1, 1, 1, 0, //
            0, -1, 0, 0, 0, 0, //
            0, -1, 0, 0, 0, 0, //
            0, -1, 0, 0, 0, 0, //
            0, 0,  0, 0, 0, 0, //
            0, 0,  0, 0, 0, 0,
        };
        auto policy =
            net->forward(torch::from_blob(board.data(), {1, 1, 6, 6}, options));

        fmt::print("Situation:\n");
        show_policy(board);

        fmt::print("policy:\n");
        show_policy(policy_from_tensor(policy.exp()));
    }
}

void dump() {
    // load net
    Net net{};
    fmt::print("Loading model and optimizer\n");
    torch::load(net, "net.pt");
    torch::optim::Adam opt(net->parameters());
    torch::load(opt, "opt.pt");
    Mcts mcts{};
    net->to(torch::kCPU);

    fmt::print("Dumping parameters\n");
    net->dump_parameters();
}
