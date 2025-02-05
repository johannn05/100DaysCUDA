#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Tile width

// tiled matmul kernel with corner turning
__global__ void matrixMulTiled(float* A, float* B, float* C, int N, int M, int K) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // iterate over tiles of A and B
    for (int t = 0; t < (M + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // load A tile (row-major access for coalescing)
        if (row < N && t * TILE_SIZE + threadIdx.x < M) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * M + t * TILE_SIZE + threadIdx.x];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile (corner turning: column-major access for coalescing)
        if (col < K && t * TILE_SIZE + threadIdx.y < M) {
            B_tile[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();  // sync to ensure tiles loaded

        // compute partial sum
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();  // sync before next tile load
    }

    // store result in C
    if (row < N && col < K) {
        C[row * K + col] = sum;
    }
}

int main() {
    int N = 64, M = 64, K = 64;  // matrices dimensions (A: N x M, B: M x K, C: N x K)
    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * K * sizeof(float);
    size_t sizeC = N * K * sizeof(float);

    float *h_A = new float[N * M];
    float *h_B = new float[M * K];
    float *h_C = new float[N * K];

    // init matrices
    for (int i = 0; i < N * M; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * K; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // launch kernel
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, M, K);
    cudaDeviceSynchronize();

    // copy result back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // print first 4x4 block of result
    std::cout << "First 4x4 block of result matrix C:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << h_C[i * K + j] << " ";
        }
        std::cout << std::endl;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
