#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

#include <torch/torch.h>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include "model.hpp"

const auto FGRED = fmt::fg(fmt::color::red);

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
    auto output = net.forward(x);

    fmt::print("input shape = {}\n"
               "output shape = {}\n",
               x.sizes(), output.sizes());
}
