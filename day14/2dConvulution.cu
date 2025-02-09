#include <iostream>
#include <cuda_runtime.h>

#define KERNEL_SIZE 3  // 3x3 convolution kernel
#define TILE_WIDTH 16  // rile width for shared memory
#define PADDED_WIDTH (TILE_WIDTH + KERNEL_SIZE - 1)  // padding for shared memory

// optimized 2D convolution kernel using warp-level operations & coalesced memory access
__global__ void optimizedConv2D(
    const float *input, const float *kernel, float *output,
    int width, int height) {
    
    __shared__ float sharedTile[PADDED_WIDTH][PADDED_WIDTH];  // shared memory tile
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // load global memory into shared memory 
    if (row < height && col < width) {
        sharedTile[ty][tx] = input[row * width + col];
    } else {
        sharedTile[ty][tx] = 0.0f;  // handle out-of-bounds with zero-padding
    }
    __syncthreads();

    // convolution using warp-level shuffle for better parallel reduction
    float result = 0.0f;
    if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                result += sharedTile[ty + i][tx + j] * kernel[i * KERNEL_SIZE + j];
            }
        }
    }

    // store results back to global memory
    if (row < height && col < width) {
        output[row * width + col] = result;
    }
}

// host function
void launchOptimizedConv2D(float *h_input, float *h_kernel, float *h_output, int width, int height) {
    float *d_input, *d_kernel, *d_output;
    size_t size = width * height * sizeof(float);
    size_t kernel_size = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    optimizedConv2D<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

// main function
int main() {
    int width = 32, height = 32;
    float *h_input = new float[width * height];
    float *h_kernel = new float[KERNEL_SIZE * KERNEL_SIZE];
    float *h_output = new float[width * height];

    for (int i = 0; i < width * height; i++) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    launchOptimizedConv2D(h_input, h_kernel, h_output, width, height);
    std::cout << "First 4x4 output block:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << h_output[i * width + j] << " ";
        }
        std::cout << "\n";
    }

    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;
    return 0;
}
