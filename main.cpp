#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#define TILE_DIM 16
#define CHECK_CADENCE 20 // Reduce PCIe traffic: evaluate convergence every 20 iterations

// Atomic max helper for single-precision floats using CAS
__device__ __forceinline__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

// -------------------------------------------------------------
// 1. GLOBAL MEMORY STENCIL KERNEL
// -------------------------------------------------------------
__global__ void heat_diffusion_global_kernel(
    const float* __restrict__ T_old,
    float* __restrict__ T_new,
    float* __restrict__ d_max_diff,
    int N,
    bool check_convergence)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float s_max[TILE_DIM * TILE_DIM];
    s_max[tid] = 0.0f;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        int idx = i * N + j;
        float top    = T_old[(i - 1) * N + j];
        float bottom = T_old[(i + 1) * N + j];
        float left   = T_old[i * N + (j - 1)];
        float right  = T_old[i * N + (j + 1)];

        float updated = 0.25f * (top + bottom + left + right);
        T_new[idx] = updated;

        if (check_convergence) {
            s_max[tid] = fabsf(updated - T_old[idx]);
        }
    }

    // In-block tree reduction for max difference
    if (check_convergence) {
        __syncthreads();
        for (int s = (TILE_DIM * TILE_DIM) / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0 && s_max[0] > 0.0f) {
            atomicMaxFloat(d_max_diff, s_max[0]);
        }
    }
}

// -------------------------------------------------------------
// 2. SHARED-MEMORY TILED STENCIL KERNEL (WITH HALO CELLS)
// -------------------------------------------------------------
__global__ void heat_diffusion_shared_kernel(
    const float* __restrict__ T_old,
    float* __restrict__ T_new,
    float* __restrict__ d_max_diff,
    int N,
    bool check_convergence)
{
    __shared__ float s_T[TILE_DIM + 2][TILE_DIM + 2];
    __shared__ float s_max[TILE_DIM * TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    s_max[tid] = 0.0f;

    int j = blockIdx.x * blockDim.x + tx;
    int i = blockIdx.y * blockDim.y + ty;

    int sm_x = tx + 1;
    int sm_y = ty + 1;

    // Load primary cell into tile
    if (i < N && j < N) {
        s_T[sm_y][sm_x] = T_old[i * N + j];
    } else {
        s_T[sm_y][sm_x] = 0.0f;
    }

    // Collaborative halo loading
    if (ty == 0) {
        s_T[0][sm_x] = (i > 0 && j < N) ? T_old[(i - 1) * N + j] : 0.0f;
    }
    if (ty == blockDim.y - 1) {
        s_T[TILE_DIM + 1][sm_x] = (i + 1 < N && j < N) ? T_old[(i + 1) * N + j] : 0.0f;
    }
    if (tx == 0) {
        s_T[sm_y][0] = (j > 0 && i < N) ? T_old[i * N + (j - 1)] : 0.0f;
    }
    if (tx == blockDim.x - 1) {
        s_T[sm_y][TILE_DIM + 1] = (j + 1 < N && i < N) ? T_old[i * N + (j + 1)] : 0.0f;
    }

    __syncthreads();

    // Compute interior cells only
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        float updated = 0.25f * (s_T[sm_y - 1][sm_x] + 
                                 s_T[sm_y + 1][sm_x] + 
                                 s_T[sm_y][sm_x - 1] + 
                                 s_T[sm_y][sm_x + 1]);
        int idx = i * N + j;
        T_new[idx] = updated;

        if (check_convergence) {
            s_max[tid] = fabsf(updated - s_T[sm_y][sm_x]);
        }
    }

    // In-block tree reduction
    if (check_convergence) {
        __syncthreads();
        for (int s = (TILE_DIM * TILE_DIM) / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0 && s_max[0] > 0.0f) {
            atomicMaxFloat(d_max_diff, s_max[0]);
        }
    }
}

// -------------------------------------------------------------
// DRIVER RUNNER FUNCTION
// -------------------------------------------------------------
void run_simulation(
    const std::vector<float>& initial_grid, 
    int N, 
    float epsilon, 
    bool use_shared_memory,
    int max_iterations = 50000)
{
    size_t bytes = N * N * sizeof(float);
    float *d_T_old = nullptr, *d_T_new = nullptr, *d_max_diff = nullptr;

    cudaMalloc(&d_T_old, bytes);
    cudaMalloc(&d_T_new, bytes);
    cudaMalloc(&d_max_diff, sizeof(float));

    cudaMemcpy(d_T_old, initial_grid.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_new, initial_grid.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int iter = 0;
    float h_max_diff = 1e9f;

    while (h_max_diff >= epsilon && iter < max_iterations) {
        bool check = ((iter + 1) % CHECK_CADENCE == 0);
        if (check) {
            cudaMemset(d_max_diff, 0, sizeof(float));
        }

        if (use_shared_memory) {
            heat_diffusion_shared_kernel<<<numBlocks, threadsPerBlock>>>(
                d_T_old, d_T_new, d_max_diff, N, check);
        } else {
            heat_diffusion_global_kernel<<<numBlocks, threadsPerBlock>>>(
                d_T_old, d_T_new, d_max_diff, N, check);
        }

        if (check) {
            cudaMemcpy(&h_max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost);
        }

        std::swap(d_T_old, d_T_new);
        iter++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    std::vector<float> final_grid(N * N);
    cudaMemcpy(final_grid.data(), d_T_old, bytes, cudaMemcpyDeviceToHost);

    std::cout << "--- " << (use_shared_memory ? "Shared-Memory Tiled" : "Global-Memory Baseline") << " ---" << std::endl;
    std::cout << "Converged after : " << iter << " iterations" << std::endl;
    std::cout << "Final max diff  : " << h_max_diff << " (tolerance: " << epsilon << ")" << std::endl;
    std::cout << "Execution time  : " << elapsed_ms << " ms" << std::endl;
    std::cout << "Center Temp T(" << N/2 << "," << N/2 << ") = " << final_grid[(N/2) * N + (N/2)] << " C\n" << std::endl;

    cudaFree(d_T_old);
    cudaFree(d_T_new);
    cudaFree(d_max_diff);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int N = 512;
    const float epsilon = 1e-4f;

    // Initialize grid with 0 C interior and fixed boundaries
    std::vector<float> grid(N * N, 0.0f);
    for (int j = 0; j < N; j++) {
        grid[0 * N + j] = 100.0f;       // Top edge = 100 C
        grid[(N - 1) * N + j] = 0.0f;   // Bottom edge = 0 C
    }
    for (int i = 0; i < N; i++) {
        grid[i * N + 0] = 50.0f;         // Left edge = 50 C
        grid[i * N + (N - 1)] = 50.0f;   // Right edge = 50 C
    }

    std::cout << "Starting 2D Heat Diffusion Benchmark (Grid: " << N << "x" << N << ")..." << std::endl;
    run_simulation(grid, N, epsilon, false); // Global memory
    run_simulation(grid, N, epsilon, true);  // Shared memory tiled

    return 0;
}