#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 32  
#define KERNEL_SIZE 5  // convolution kernel size
#define PADDED_TILE_SIZE (TILE_SIZE + KERNEL_SIZE - 1)  // Avoids bank conflicts

// optimized 1D convolution kernel with shared memory
__global__ void conv1D(float *input, float *kernel, float *output, int N) {
    __shared__ float tile[PADDED_TILE_SIZE];  

    int tx = threadIdx.x;
    int idx = blockIdx.x * TILE_SIZE + tx;  // global index in input array

    // load data into shared memory with padding
    int halo_index_left = idx - KERNEL_SIZE / 2;
    if (halo_index_left >= 0 && halo_index_left < N) {
        tile[tx] = input[halo_index_left];
    } else {
        tile[tx] = 0.0f;  // zero padding
    }

    __syncthreads();  // ensure all threads load before computation

    // compute 1D convolution
    float sum = 0.0f;
    if (tx < TILE_SIZE && idx < N) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            sum += tile[tx + i] * kernel[i];
        }
        output[idx] = sum;
    }
}

// host function to launch kernel
void run1DConvolution(float *input, float *kernel, float *output, int N) {
    float *d_input, *d_kernel, *d_output;
    size_t inputSize = N * sizeof(float);
    size_t kernelSize = KERNEL_SIZE * sizeof(float);

    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_kernel, kernelSize);
    cudaMalloc((void**)&d_output, inputSize);

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE);

    conv1D<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, inputSize, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

// 
void initializeArray(float *A, int size) {
    for (int i = 0; i < size; i++)
        A[i] = static_cast<float>(rand()) / RAND_MAX;
}

// main 
int main() {
    int N = 128;

    float *input = new float[N];
    float *kernel = new float[KERNEL_SIZE];
    float *output = new float[N];

    initializeArray(input, N);
    initializeArray(kernel, KERNEL_SIZE);
    run1DConvolution(input, kernel, output, N);

    std::cout << "First 10 values of output:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    delete[] input;
    delete[] kernel;
    delete[] output;

    return 0;
}
