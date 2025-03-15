#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 3  // matrix size

// function to perform matrix addition using cublas
void cublasMatrixAdd(float *h_A, float *h_B, float *h_C, int n) {
    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // initialize cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // perform C = A + B using cublasSaxpy (y = alpha * x + y)
    float alpha = 1.0f;
    cublasSaxpy(handle, n * n, &alpha, d_A, 1, d_B, 1);

    cudaMemcpy(h_C, d_B, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// main function
int main() {
    float h_A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float h_B[N * N] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_C[N * N] = {0};

    cublasMatrixAdd(h_A, h_B, h_C, N);

    std::cout << "matrix C (A + B):" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
