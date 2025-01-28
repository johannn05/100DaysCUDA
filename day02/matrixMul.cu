#include <iostream>
#include <cuda_runtime.h>

// matmul kernel
__global__ void matrixMultiply(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row idx of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col idx of C

    if (row < rowsA && col < colsB) {
        float value = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            value += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = value;
    }
}

int main() {
    // dims
    int rowsA = 4, colsA = 3, rowsB = 3, colsB = 4;
    int rowsC = rowsA, colsC = colsB;

    float *h_A = new float[rowsA * colsA];
    float *h_B = new float[rowsB * colsB];
    float *h_C = new float[rowsC * colsC];

    // initialize host matrices
    for (int i = 0; i < rowsA * colsA; ++i) h_A[i] = i + 1; 
    for (int i = 0; i < rowsB * colsB; ++i) h_B[i] = i + 1; 

    float *d_A, *d_B, *d_C;

    // allocate device memory
    cudaMalloc((void **)&d_A, rowsA * colsA * sizeof(float));
    cudaMalloc((void **)&d_B, rowsB * colsB * sizeof(float));
    cudaMalloc((void **)&d_C, rowsC * colsC * sizeof(float));

    // copy host matrices to device
    cudaMemcpy(d_A, h_A, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((colsC + blockDim.x - 1) / blockDim.x, 
                 (rowsC + blockDim.y - 1) / blockDim.y);

    // kernel launch
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // copy result to host
    cudaMemcpy(h_C, d_C, rowsC * colsC * sizeof(float), cudaMemcpyDeviceToHost);

    // print result matrix
    std::cout << "Result C:" << std::endl;
    for (int i = 0; i < rowsC; ++i) {
        for (int j = 0; j < colsC; ++j) {
            std::cout << h_C[i * colsC + j] << " ";
        }
        std::cout << std::endl;
    }

    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
