cmake_minimum_required(VERSION 3.28)
project(untitled CUDA C CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(untitled main.cu
        headers/parameters.h
        headers/numeric_functions.h
        headers/init_env.h
        headers/agent_update.h
        headers/update_matrices.h
        headers/logging.h
        headers/gaussian_odour.h
        headers/beta_sampling.h
        main.cu
        headers/agent_update.h
)

set_target_properties(untitled PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(untitled stdc++)

target_link_libraries(untitled m)
