#pragma once

#include "game.hpp"
#include "model.hpp"
#include "tensor_utils.hpp"

class NetQuery {
  public:
    NetQuery(Net net);
    std::pair<Action, std::array<float, 36>> raw_query(State state);

  private:
    Net net;
};
