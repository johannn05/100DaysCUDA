#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16  

// tiled matmul kernel
__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, int N, int M, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (M + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // load tiles into shared memory
        if (row < N && (t * TILE_SIZE + threadIdx.x) < M) 
            tileA[threadIdx.y][threadIdx.x] = A[row * M + t * TILE_SIZE + threadIdx.x];
        else 
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && (t * TILE_SIZE + threadIdx.y) < M) 
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else 
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // sync threads to ensure tiles are fully loaded

        // do computation within tile
        for (int k = 0; k < TILE_SIZE; ++k) 
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();  // ensure all threads done before loading next tile
    }

    // store result in global memory
    if (row < N && col < K)
        C[row * K + col] = sum;
}

int main() {
    int N = 4, M = 3, K = 4;  // matrix dims: A (N x M), B (M x K), C (N x K)
    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * K * sizeof(float);
    size_t sizeC = N * K * sizeof(float);

    float *h_A = new float[N * M]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // A (4x3)
    float *h_B = new float[M * K]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // B (3x4)
    float *h_C = new float[N * K](); // result C (4x4), init with 0s

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // define grid and block dims
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // kernel launch
    matrixMultiplyTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, M, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "Result C:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << h_C[i * K + j] << " ";
        }
        std::cout << std::endl;
    }

    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
