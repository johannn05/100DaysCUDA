#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8  

// prefix sum kernel with shared memory
__global__ void prefixSumShared(float* d_in, float* d_out, int n) {
    __shared__ float temp[BLOCK_SIZE];  // shared memory for block

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx < n) temp[tid] = d_in[idx];  // load into shared memory
    __syncthreads();  // ensure all threads have loaded data

    // hillis-steele scan
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        float val = 0;
        if (tid >= stride) val = temp[tid - stride];  // read left neighbor
        __syncthreads();
        temp[tid] += val;  // in-place accumulation
        __syncthreads();  // sync before next step
    }

    if (idx < n) d_out[idx] = temp[tid];  // store result in global memory
}

int main() {
    int n = BLOCK_SIZE;
    size_t size = n * sizeof(float);
    
    float h_in[] = {1, 2, 3, 4, 5, 6, 7, 8}; 
    float* h_out = new float[n](); // output array, dynamically allocated

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // kernel launch
    prefixSumShared<<<1, BLOCK_SIZE>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "Prefix Sum output: ";
    for (int i = 0; i < n; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // free memory
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
