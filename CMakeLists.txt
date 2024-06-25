cmake_minimum_required(VERSION 3.28)
project(RayTracing)

set(CMAKE_CXX_STANDARD 17)

add_executable(RayTracing main.cpp
        vec3.h)
add_executable(BinaryRadixTree binary_radix_tree.cpp)
