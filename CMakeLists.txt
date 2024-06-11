cmake_minimum_required(VERSION 3.28)
project(RayTracing CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(RayTracing main.cu)

set_target_properties(
        RayTracing
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)