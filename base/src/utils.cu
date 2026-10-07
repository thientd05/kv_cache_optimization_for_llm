#include "utils.cuh"

#include <cstdlib>
#include <iostream>

#include "config.h"

std::string g_model_dir = DEFAULT_MODEL_DIR;

std::string modelFilePath(const std::string &filename)
{
    if (g_model_dir.empty())
        return filename;
    if (g_model_dir.back() == '/')
        return g_model_dir + filename;
    return g_model_dir + "/" + filename;
}

void *allocDevice(size_t bytes, const char *what)
{
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Fatal error: failed to allocate " << bytes / B_TO_MB << " MiB for "
                  << what << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    return ptr;
}

int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cerr << "Device: " << prop.name << "\n";
    std::cerr << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cerr << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cerr << "SM count: " << prop.multiProcessorCount << "\n";
    std::cerr << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cerr << "Free memory: " << free_mem / 1024 / 1024 << " MB, total memory: " << total_mem / 1024 / 1024 << " MB\n";
    return 0;
}
