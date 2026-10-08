cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")
set(CUDACXX "/usr/bin/nvcc")

project(tiny-vllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")

find_package(CUDAToolkit REQUIRED)

add_executable(tiny-vllm
    src/main.cu
    src/engine.cu
    src/kernels.cu
    src/model.cu
    src/utils.cu
    src/request_queue.cpp
)

target_include_directories(tiny-vllm PRIVATE src)
target_include_directories(tiny-vllm PRIVATE include)

# Knobs for the §7.2 block-size ablation. sweep.py rebuilds with -DABLATION_BLOCK_SIZE=N;
# nothing else ever sets them, so a normal build is unaffected.
if(DEFINED ABLATION_BLOCK_SIZE)
    target_compile_definitions(tiny-vllm PRIVATE ABLATION_BLOCK_SIZE=${ABLATION_BLOCK_SIZE})
endif()
if(DEFINED ABLATION_MAX_SEQUENCES)
    target_compile_definitions(tiny-vllm PRIVATE ABLATION_MAX_SEQUENCES=${ABLATION_MAX_SEQUENCES})
endif()

target_link_libraries(tiny-vllm PRIVATE
    CUDA::cublas
    CUDA::cudart
)
