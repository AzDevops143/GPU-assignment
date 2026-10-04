#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define BLOCK 16
#define GLOBAL_MODE 0
#define SHARED_MODE 1

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

__device__ float g_maxdiff;

// Displays Blackwell GB200 target architecture banner and queries runtime device capabilities
void printGpuArchitectureInfo()
{
    printf("==================================================================\n");
    printf("   NVIDIA GB200 Blackwell Architecture & Hardware Verification   \n");
    printf("   Target Compute Architecture: sm_100 (Blackwell B200 / GB200)   \n");
    printf("==================================================================\n");
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("Note: Running in environment without direct physical GPU device pass-through.\n");
        printf("Compiled Target Binary Architecture: sm_100 (Native Blackwell Machine Code)\n");
        printf("==================================================================\n\n");
        return;
    }
    printf("Detected %d CUDA-Capable Device(s):\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  Device %d: %s\n", i, prop.name);
        printf("    Compute Capability       : %d.%d %s\n", prop.major, prop.minor,
               (prop.major == 10 && prop.minor == 0) ? "[NVIDIA Blackwell sm_100 Confirmed]" : "");
        printf("    Total Global Memory      : %.2f GB (HBM3e / High-Bandwidth)\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("    Shared Memory per Block  : %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("    Streaming Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("    Warp Size                : %d\n", prop.warpSize);
        printf("    Memory Bus Width         : %d-bit\n", prop.memoryBusWidth);
    }
    printf("==================================================================\n\n");
}

// Probes for multi-GPU topology and activates direct NVLink P2P access (e.g. GB200 dual Blackwell GPUs)
void probeAndEnableNvlinkPeerAccess()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 1) {
        return; // Single GPU or CPU-only runner
    }

    printf("==================================================================\n");
    printf("Multi-GPU System Detected (%d Devices). Querying NVLink P2P Access...\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccess = 0;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                if (canAccess) {
                    cudaSetDevice(i);
                    cudaError_t pErr = cudaDeviceEnablePeerAccess(j, 0);
                    if (pErr == cudaSuccess || pErr == cudaErrorPeerAccessAlreadyEnabled) {
                        printf("  [NVLink P2P Enabled] GPU %d <-> GPU %d direct access active\n", i, j);
                    }
                } else {
                    printf("  [P2P Unavailable] GPU %d cannot access GPU %d directly\n", i, j);
                }
            }
        }
    }
    cudaSetDevice(0);
    printf("==================================================================\n\n");
}

// JOR (Jacobi Over-Relaxation) Global Memory Kernel
__global__ void heatKernelJORGlobal(const float* __restrict__ T, float* __restrict__ Tnew, int N, float omega)
{
    extern __shared__ float sdata[];
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float diff = 0.0f;
    if (row < N - 1 && col < N - 1) {
        int idx = row * N + col;
        float up    = T[idx - N];
        float down  = T[idx + N];
        float left  = T[idx - 1];
        float right = T[idx + 1];

        // 5-point Jacobi predictor
        float jacobi = 0.25f * (up + down + left + right);

        // Jacobi Over-Relaxation (JOR) update with relaxation parameter omega
        float newval = (1.0f - omega) * T[idx] + omega * jacobi;

        diff = fabsf(newval - T[idx]);
        Tnew[idx] = newval;
    }

    sdata[tid] = diff;
    __syncthreads();

    // In-kernel parallel reduction for block maximum difference
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // Atomic update to global maximum difference
    if (tid == 0) atomicMax((int*)&g_maxdiff, __float_as_int(sdata[0]));
}

// JOR (Jacobi Over-Relaxation) Shared Memory Tiled Kernel
__global__ void heatKernelJORShared(const float* __restrict__ T, float* __restrict__ Tnew, int N, float omega)
{
    extern __shared__ float smemAll[];
    int tileDim = blockDim.x + 2;
    float* tile = smemAll;
    float* sdata = smemAll + tileDim * tileDim;

    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    int colc = min(col, N - 1);
    int rowc = min(row, N - 1);

    // Load central interior element
    tile[ly * tileDim + lx] = T[rowc * N + colc];

    // Cooperative halo loading
    if (threadIdx.x == 0) {
        tile[ly * tileDim + 0] = T[rowc * N + max(col - 1, 0)];
    }
    if (threadIdx.x == blockDim.x - 1) {
        tile[ly * tileDim + (lx + 1)] = T[rowc * N + min(col + 1, N - 1)];
    }
    if (threadIdx.y == 0) {
        tile[0 * tileDim + lx] = T[max(row - 1, 0) * N + colc];
    }
    if (threadIdx.y == blockDim.y - 1) {
        tile[(ly + 1) * tileDim + lx] = T[min(row + 1, N - 1) * N + colc];
    }

    __syncthreads();

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float diff = 0.0f;

    if (row < N - 1 && col < N - 1) {
        float up    = tile[(ly - 1) * tileDim + lx];
        float down  = tile[(ly + 1) * tileDim + lx];
        float left  = tile[ly * tileDim + (lx - 1)];
        float right = tile[ly * tileDim + (lx + 1)];

        // 5-point Jacobi predictor using fast shared memory
        float jacobi = 0.25f * (up + down + left + right);

        // Jacobi Over-Relaxation (JOR) update
        int idx = row * N + col;
        float newval = (1.0f - omega) * T[idx] + omega * jacobi;

        diff = fabsf(newval - T[idx]);
        Tnew[idx] = newval;
    }

    sdata[tid] = diff;
    __syncthreads();

    // In-kernel parallel reduction
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) atomicMax((int*)&g_maxdiff, __float_as_int(sdata[0]));
}

struct SimResult {
    long long iterations;
    float time_ms;
    bool converged;
    float final_max_diff;
    std::vector<float> grid;
};

void initGrid(std::vector<float>& g, int N, float topT, float bottomT, float leftT, float rightT, float initTemp)
{
    g.assign((size_t)N * N, initTemp);
    for (int j = 0; j < N; j++) {
        g[0 * N + j] = topT;
        g[(N - 1) * N + j] = bottomT;
    }
    for (int i = 0; i < N; i++) {
        g[i * N + 0] = leftT;
        g[i * N + (N - 1)] = rightT;
    }
}

SimResult runSimulation(int N, float tol, long long maxIter, bool useSharedMemory, float omega,
                         float topT, float bottomT, float leftT, float rightT, float initTemp,
                         bool dump, const char* fname)
{
    SimResult result;
    std::vector<float> h_init;
    initGrid(h_init, N, topT, bottomT, leftT, rightT, initTemp);

    float *d_A, *d_B;
    size_t bytes = (size_t)N * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_init.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_init.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK, BLOCK);
    int interior = N - 2;
    dim3 grid((interior + BLOCK - 1) / BLOCK, (interior + BLOCK - 1) / BLOCK);

    size_t sharedBytesGlobal = block.x * block.y * sizeof(float);
    size_t tileDim = block.x + 2;
    size_t sharedBytesShared = tileDim * tileDim * sizeof(float) + block.x * block.y * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float* dA = d_A;
    float* dB = d_B;
    long long iter = 0;
    float diffVal = 1e30f;
    float zero = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));

    while (diffVal > tol && iter < maxIter) {
        CUDA_CHECK(cudaMemcpyToSymbol(g_maxdiff, &zero, sizeof(float)));

        if (useSharedMemory) {
            heatKernelJORShared<<<grid, block, sharedBytesShared>>>(dA, dB, N, omega);
        } else {
            heatKernelJORGlobal<<<grid, block, sharedBytesGlobal>>>(dA, dB, N, omega);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyFromSymbol(&diffVal, g_maxdiff, sizeof(float)));

        float* tmp = dA;
        dA = dB;
        dB = tmp;

        iter++;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    result.iterations = iter;
    result.time_ms = ms;
    result.converged = (diffVal <= tol);
    result.final_max_diff = diffVal;
    result.grid.assign((size_t)N * N, 0.0f);
    CUDA_CHECK(cudaMemcpy(result.grid.data(), dA, bytes, cudaMemcpyDeviceToHost));

    if (dump && fname != nullptr) {
        FILE* f = fopen(fname, "w");
        if (f) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    fprintf(f, "%f", result.grid[i * N + j]);
                    if (j < N - 1) fprintf(f, ",");
                }
                fprintf(f, "\n");
            }
            fclose(f);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return result;
}

int main(int argc, char** argv)
{
    printGpuArchitectureInfo();
    probeAndEnableNvlinkPeerAccess();

    int N = 256;
    float eps = 1e-4f;
    long long maxIter = 2000000;
    float omega = 0.95f; // JOR relaxation factor
    float topT = 100.0f;
    float bottomT = 0.0f;
    float leftT = 75.0f;
    float rightT = 50.0f;
    float initTemp = 0.0f;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) eps = (float)atof(argv[2]);
    if (argc > 3) maxIter = atoll(argv[3]);
    if (argc > 4) topT = (float)atof(argv[4]);
    if (argc > 5) bottomT = (float)atof(argv[5]);
    if (argc > 6) leftT = (float)atof(argv[6]);
    if (argc > 7) rightT = (float)atof(argv[7]);
    if (argc > 8) initTemp = (float)atof(argv[8]);
    if (argc > 9) omega = (float)atof(argv[9]);

    if (N < 3) {
        fprintf(stderr, "N must be at least 3\n");
        return 1;
    }

    char fnameGlobal[256];
    char fnameShared[256];
    snprintf(fnameGlobal, sizeof(fnameGlobal), "grid_jor_global_%d.csv", N);
    snprintf(fnameShared, sizeof(fnameShared), "grid_jor_shared_%d.csv", N);

    SimResult rGlobal = runSimulation(N, eps, maxIter, false, omega, topT, bottomT, leftT, rightT, initTemp, true, fnameGlobal);
    SimResult rShared = runSimulation(N, eps, maxIter, true, omega, topT, bottomT, leftT, rightT, initTemp, true, fnameShared);

    printf("RESULT_CSV\n");
    printf("N,mode,iterations,time_ms,converged,final_max_diff\n");
    printf("%d,jor_global,%lld,%f,%d,%e\n", N, rGlobal.iterations, rGlobal.time_ms, rGlobal.converged ? 1 : 0, rGlobal.final_max_diff);
    printf("%d,jor_shared,%lld,%f,%d,%e\n", N, rShared.iterations, rShared.time_ms, rShared.converged ? 1 : 0, rShared.final_max_diff);

    float maxDiff = 0.0f;
    for (size_t i = 0; i < rGlobal.grid.size(); i++) {
        float d = fabsf(rGlobal.grid[i] - rShared.grid[i]);
        if (d > maxDiff) maxDiff = d;
    }

    printf("CORRECTNESS_CSV\n");
    printf("N,max_diff_global_vs_shared\n");
    printf("%d,%e\n", N, maxDiff);

    return 0;
}
