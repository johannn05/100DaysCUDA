#include <iostream>
#include <cuda_runtime.h>

// kernel function
__global__ void matrixAdd(const float *A, const float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
    }
}

int main() {
    int rows = 4, cols = 4;
    int size = rows * cols * sizeof(float);

    // initialize host matrices
    float h_A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_B[] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_C[rows * cols];

    float *d_A, *d_B, *d_C;

    // allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // define block and grid dims
    dim3 blockDim(2, 2);  // 2 by 2 threads per block
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, 
                 (rows + blockDim.y - 1) / blockDim.y); 

    // launch kernel
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // copy result to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // print result
    std::cout << "Result C:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_C[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
