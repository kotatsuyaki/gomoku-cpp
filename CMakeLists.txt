cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(main)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Use mold linker if available
find_program(MOLD mold)
if (NOT ${MOLD} STREQUAL "MOLD-NOTFOUND")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --ld-path=${MOLD}")
endif()

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main "${CUDA_LIBRARIES}")
# target_link_libraries(main "-Wl,--no-as-needed -ltorch_cuda")
set_property(TARGET main PROPERTY CXX_STANDARD 14)

# For development
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

