#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16   // tile size for shared memory optimization
#define KERNEL_SIZE 3   // 3x3 convolution kernel
#define PADDING 1       // padding to keep output size same
#define STRIDE 1        

// ptimized CNN forward pass kernel
__global__ void conv2DForward(float* input, float* filter, float* output, int inH, int inW, int outH, int outW, int numFilters) {
    __shared__ float tile[BLOCK_SIZE + KERNEL_SIZE - 1][BLOCK_SIZE + KERNEL_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    int sharedSize = BLOCK_SIZE + KERNEL_SIZE - 1;
    int halfKernel = KERNEL_SIZE / 2;

    // load tile into shared memory (input region for convolution)
    if (row < inH && col < inW) {
        tile[ty][tx] = input[row * inW + col];
    } else {
        tile[ty][tx] = 0.0f; // Zero-padding
    }
    __syncthreads();

    if (ty < BLOCK_SIZE && tx < BLOCK_SIZE && row < outH && col < outW) {
        for (int f = 0; f < numFilters; ++f) {
            float sum = 0.0f;

            // apply convolution filter
            for (int i = 0; i < KERNEL_SIZE; ++i) {
                for (int j = 0; j < KERNEL_SIZE; ++j) {
                    sum += tile[ty + i][tx + j] * filter[f * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j];
                }
            }

            output[(f * outH * outW) + (row * outW + col)] = sum;
        }
    }
}

void runConv2D(float* input, float* filter, float* output, int inH, int inW, int outH, int outW, int numFilters) {
    float* d_input, * d_filter, * d_output;
    int inputSize = inH * inW * sizeof(float);
    int filterSize = numFilters * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    int outputSize = outH * outW * numFilters * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_filter, filterSize);
    cudaMalloc(&d_output, outputSize);
    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((outW + BLOCK_SIZE - 1) / BLOCK_SIZE, (outH + BLOCK_SIZE - 1) / BLOCK_SIZE);

    conv2DForward<<<gridDim, blockDim>>>(d_input, d_filter, d_output, inH, inW, outH, outW, numFilters);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
}

void randomInit(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX * 2 - 1;  // Range [-1,1]
    }
}

int main() {
    int inH = 32, inW = 32;
    int outH = inH, outW = inW;  // same output size with padding
    int numFilters = 8;

    float* input = new float[inH * inW];
    float* filter = new float[numFilters * KERNEL_SIZE * KERNEL_SIZE];
    float* output = new float[outH * outW * numFilters];

    randomInit(input, inH * inW);
    randomInit(filter, numFilters * KERNEL_SIZE * KERNEL_SIZE);

    runConv2D(input, filter, output, inH, inW, outH, outW, numFilters);

    std::cout << "First 4x4 block of output:" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << output[i * outW + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] input;
    delete[] filter;
    delete[] output;

    return 0;
}
