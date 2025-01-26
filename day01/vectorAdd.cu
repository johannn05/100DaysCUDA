#include <iostream>

__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000;
    int size = N * sizeof(float);
    
    // host arrays
    float *A = new float[N], *B = new float[N], *C = new float[N];

    // define A and B
    for (int i = 0; i < N; ++i) {
        A[i] = i * 1.0f;       
        B[i] = i * 2.0f;       // double of A
    }

    // device arrays
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // launch kernel, ceiling division for number of blocks
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // print first 10 results
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }

    // free memory
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
