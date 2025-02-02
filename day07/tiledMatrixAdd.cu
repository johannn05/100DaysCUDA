#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // tile size

// tiled matrix addition kernel
__global__ void matrixAddTiled(float* A, float* B, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory for tiles
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    // load tiles into shared memory
    if (row < numRows && col < numCols) {
        A_tile[localRow][localCol] = A[row * numCols + col];
        B_tile[localRow][localCol] = B[row * numCols + col];
    } else {
        A_tile[localRow][localCol] = 0.0f;
        B_tile[localRow][localCol] = 0.0f;
    }
    __syncthreads();  // sync threads before computation

    // element-wise addition
    if (row < numRows && col < numCols) {
        C[row * numCols + col] = A_tile[localRow][localCol] + B_tile[localRow][localCol];
    }
}

int main() {
    int numRows = 4, numCols = 4;
    size_t size = numRows * numCols * sizeof(float);

    float* h_A = new float[numRows * numCols]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float* h_B = new float[numRows * numCols]{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float* h_C = new float[numRows * numCols]();  // dynamically allocate and initialize to 0

    // device pointers
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((numCols + TILE_WIDTH - 1) / TILE_WIDTH, 
                 (numRows + TILE_WIDTH - 1) / TILE_WIDTH);

    // kernel launch
    matrixAddTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, numRows, numCols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "resultant matrix C:" << std::endl;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cout << h_C[i * numCols + j] << " ";
        }
        std::cout << std::endl;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
