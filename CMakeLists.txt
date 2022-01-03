cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(main)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(fmt REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main src/main.cpp src/model.cpp src/mcts.cpp src/game.cpp src/net_query.cpp ./src/tensor_utils.cpp)

# Link libraries
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main "${CUDA_LIBRARIES}")
target_link_libraries(main fmt::fmt)

set_property(TARGET main PROPERTY CXX_STANDARD 17)

# Enable OMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(main OpenMP::OpenMP_CXX)
endif()

# For development
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

