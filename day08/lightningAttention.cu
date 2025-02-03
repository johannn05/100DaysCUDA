#include <iostream>
#include <cuda_runtime.h>

#define B 16  // block size
#define D 64  // feature dim
#define EPSILON 1e-5  // Small constant for numerical stability

// lightning attention kernel
__global__ void lightningAttention(float* Q, float* K, float* V, float* O, float* M, int n, int d) {
    __shared__ float Q_block[B * D];
    __shared__ float K_block[B * D];
    __shared__ float V_block[B * D];
    __shared__ float KV[D * D];

    int t = blockIdx.x;  
    int tid = threadIdx.x;  

    int offset = t * B * d;

    // load Q, K, V blocks from global memory to shared memory
    for (int i = tid; i < B * D; i += blockDim.x) {
        if (offset + i < n * d) {
            Q_block[i] = Q[offset + i];
            K_block[i] = K[offset + i];
            V_block[i] = V[offset + i];
        } else {
            Q_block[i] = 0.0f;
            K_block[i] = 0.0f;
            V_block[i] = 0.0f;
        }
    }
    __syncthreads();

    // initialize KV = 0 in shared memory
    for (int i = tid; i < D * D; i += blockDim.x) {
        KV[i] = 0.0f;
    }
    __syncthreads();

    // compute O_intra = [(Q * K^T) ⊙ M] * V (parallelized)
    __shared__ float O_intra[B * D];
    for (int i = tid; i < B * D; i += blockDim.x) {
        O_intra[i] = 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < B * B; i += blockDim.x) {
        int row = i / B, col = i % B;
        float dot_product = 0.0f;
        for (int k = 0; k < D; k++) {
            dot_product += Q_block[row * D + k] * K_block[col * D + k];
        }
        dot_product *= M[row * B + col];  // Apply mask
        for (int k = 0; k < D; k++) {
            atomicAdd(&O_intra[row * D + k], dot_product * V_block[col * D + k]);
        }
    }
    __syncthreads();

    // compute O_inter = Q * (KV^T) using parallel reduction
    for (int i = tid; i < D * D; i += blockDim.x) {
        int row = i / D, col = i % D;
        atomicAdd(&KV[row * D + col], K_block[row * D + col] * V_block[row * D + col]);
    }
    __syncthreads();
    __shared__ float O_inter[B * D];
    for (int i = tid; i < B * D; i += blockDim.x) {
        O_inter[i] = 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < B * D; i += blockDim.x) {
        int row = i / D, col = i % D;
        float dot_product = 0.0f;
        for (int k = 0; k < D; k++) {
            dot_product += Q_block[row * D + k] * KV[k * D + col];
        }
        O_inter[row * D + col] = dot_product;
    }
    __syncthreads();

    // store O_t = O_intra + O_inter back to global memory
    for (int i = tid; i < B * D; i += blockDim.x) {
        if (offset + i < n * d) {
            O[offset + i] = O_intra[i] + O_inter[i];
        }
    }
}

int main() {
    int n = 128;  // num tokens
    int d = D;  

    size_t size = n * d * sizeof(float);
    size_t mask_size = B * B * sizeof(float);

    // allocate host mem
    float *h_Q = new float[n * d];
    float *h_K = new float[n * d];
    float *h_V = new float[n * d];
    float *h_O = new float[n * d];
    float *h_M = new float[B * B];

    // init with random values
    for (int i = 0; i < n * d; i++) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < B * B; i++) {
        h_M[i] = (i % B <= i / B) ? 1.0f : 0.0f;  // mask condition
    }

    // allocate device mem
    float *d_Q, *d_K, *d_V, *d_O, *d_M;
    cudaMalloc((void**)&d_Q, size);
    cudaMalloc((void**)&d_K, size);
    cudaMalloc((void**)&d_V, size);
    cudaMalloc((void**)&d_O, size);
    cudaMalloc((void**)&d_M, mask_size);

    // copy data to device
    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, mask_size, cudaMemcpyHostToDevice);

    int num_blocks = n / B;
    dim3 blockDim(256);
    dim3 gridDim(num_blocks);

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record start event
    cudaEventRecord(start);

    // launch kernel
    lightningAttention<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_M, n, d);
    cudaEventRecord(stop);

    // sync and calculate elapsed time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result to host
    cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost);

    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "First 8 values of O:" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << h_O[i] << " ";
    }
    std::cout << std::endl;

    // free memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_M);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
