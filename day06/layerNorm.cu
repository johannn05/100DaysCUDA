#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-5
#define N 8  // num elements in layer

// naive layernorm kernel
__global__ void layerNormKernel(float* d_in, float* d_out, float* d_gamma, float* d_beta, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    // compute mean
    float mean = 0.0f;
    for (int i = 0; i < n; i++) {
        mean += d_in[i];
    }
    mean /= n;

    //compute variance
    float variance = 0.0f;
    for (int i = 0; i < n; i++) {
        variance += (d_in[i] - mean) * (d_in[i] - mean);
    }
    variance /= n;

    //normalize and apply scale/shift
    d_out[idx] = ((d_in[idx] - mean) / sqrtf(variance + EPSILON)) * d_gamma[idx] + d_beta[idx];
}

int main() {
    float h_in[N] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};  
    float h_gamma[N] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // scale
    float h_beta[N] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // shift
    float h_out[N] = {0}; 

    float *d_in, *d_out, *d_gamma, *d_beta;
    size_t size = N * sizeof(float);
    
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMalloc((void**)&d_gamma, size);
    cudaMalloc((void**)&d_beta, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, size, cudaMemcpyHostToDevice);

    // launch kernel
    layerNormKernel<<<1, N>>>(d_in, d_out, d_gamma, d_beta, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    std::cout << "layernorm output: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}
