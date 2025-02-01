#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define N 8  // num elements

// naive softmax kernel
__global__ void softmaxKernel(float* d_in, float* d_out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    // compute exponentials
    float exp_val = expf(d_in[idx]);

    // compute sum of exponentials (naive approach: each thread will recompute)
    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_exp += expf(d_in[i]);
    }

    // compute softmax
    d_out[idx] = exp_val / sum_exp;
}

int main() {
    float h_in[N] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};  
    float h_out[N] = {0};  

    float *d_in, *d_out;
    size_t size = N * sizeof(float);
    
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // launch kernel (1 block, N threads)
    softmaxKernel<<<1, N>>>(d_in, d_out, N);
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
