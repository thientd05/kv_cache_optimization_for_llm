cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")
project(attn-micro LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")

find_package(CUDAToolkit REQUIRED)

# Two binaries, each linking the kernels of its own source tree. Nothing is copied, so the
# microbenchmark can never drift from the engine it is measuring; separate binaries keep the
# identically-named symbols in the two trees from colliding.
foreach(variant base paged)
    if(variant STREQUAL "base")
        set(tree "${CMAKE_CURRENT_SOURCE_DIR}/../base/src")
    else()
        set(tree "${CMAKE_CURRENT_SOURCE_DIR}/../paged_attention/src")
    endif()
    add_executable(attn_bench_${variant} main_${variant}.cu ${tree}/kernels.cu)
    target_include_directories(attn_bench_${variant} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${tree})
    target_link_libraries(attn_bench_${variant} PRIVATE CUDA::cudart)
endforeach()
