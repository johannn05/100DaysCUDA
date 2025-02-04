#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define N 8  // num elements
#define BLOCK_SIZE 8  

// optimized softmax kernel with shared memory
__global__ void softmaxKernel(float* d_in, float* d_out, int n) {
    __shared__ float exp_vals[BLOCK_SIZE];  
    __shared__ float sum_exp;  

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx >= n) return;

    // compute exponentials and store in shared memory
    exp_vals[tid] = expf(d_in[idx]);
    __syncthreads();

    // parallel reduction to compute sum of exponentials
    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum_exp += exp_vals[i];
        }
    }
    __syncthreads();  // ensure sum_exp is updated before use

    // compute softmax
    d_out[idx] = exp_vals[tid] / sum_exp;
}

int main() {
    float h_in[N] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};  
    float h_out[N] = {0};  

    float *d_in, *d_out;
    size_t size = N * sizeof(float);
    
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // launch kernel 
    softmaxKernel<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "softmax output: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
