#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

std::vector<float> generate_random_array(size_t size) {
    std::vector<float> array(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        array[i] = dist(gen);
    }
    return array;
}

__global__ void sum_kernel_reduction(const float* array, float* result, int size) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (idx < size) ? array[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

float parallel_sum(const std::vector<float>& array, int block_size) {
    float* d_array = nullptr;
    float* d_result = nullptr;
    float h_result = 0.0f;

    checkCudaError(cudaMalloc(&d_array, array.size() * sizeof(float)), "cudaMalloc d_array");
    checkCudaError(cudaMalloc(&d_result, sizeof(float)), "cudaMalloc d_result");

    checkCudaError(cudaMemcpy(d_array, array.data(), array.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D array");
    checkCudaError(cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D result");

    int grid_size = (array.size() + block_size - 1) / block_size;
    sum_kernel_reduction << <grid_size, block_size, block_size * sizeof(float) >> > (d_array, d_result, array.size());

    checkCudaError(cudaGetLastError(), "Kernel execution");
    checkCudaError(cudaDeviceSynchronize(), "Device sync");

    checkCudaError(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H result");

    cudaFree(d_array);
    cudaFree(d_result);

    return h_result;
}

int main(int argc, char** argv) {
    const size_t array_size = 10000000;
    const int num_runs = 100;
    const int block_size = 1024; // Оптимальный размер блока для большинства GPU
    double total_time = 0.0;
    float reference_sum = 0.0f;

    auto array = generate_random_array(array_size);

    // Первый запуск (без учета в среднее)
    auto start = std::chrono::high_resolution_clock::now();
    reference_sum = parallel_sum(array, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Reference sum: " << reference_sum << std::endl;
    std::cout << "First run time: " << duration.count() << " seconds\n\n";

    // Основные запуски (100 раз)
    for (int i = 0; i < num_runs; ++i) {
        start = std::chrono::high_resolution_clock::now();
        float sum = parallel_sum(array, block_size);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        total_time += duration.count();

    }

    // Вывод результатов
    double avg_time = total_time / num_runs;
    std::cout << "Block size: " << block_size << std::endl;
    std::cout << "Average time over " << num_runs << " runs: " << avg_time << " seconds" << std::endl;
    std::cout << "Total time: " << total_time << " seconds" << std::endl;

    return 0;
}