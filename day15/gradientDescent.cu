#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <ctime>

#define BLOCK_SIZE 256  // Tuned for performance

// kernel for gradient descent
__global__ void gradientDescentKernel(float *X, float *y, float *theta, float *grad, int num_samples, int num_features, float lr) {
    __shared__ float grad_shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_grad = 0.0f;

    if (idx < num_samples) {
        float pred = 0.0f;
        for (int j = 0; j < num_features; j++) {
            pred += X[idx * num_features + j] * theta[j];
        }

        float error = pred - y[idx];

        for (int j = 0; j < num_features; j++) {
            local_grad += error * X[idx * num_features + j];
        }
    }

    // shared memory
    grad_shared[tid] = local_grad;
    __syncthreads();

    // block-wise parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            grad_shared[tid] += grad_shared[tid + stride];
        }
        __syncthreads();
    }

    // store the final reduced value in global memory
    if (tid == 0) {
        atomicAdd(grad, grad_shared[0]);
    }

    __syncthreads();

    if (idx == 0) {
        for (int j = 0; j < num_features; j++) {
            theta[j] -= lr * grad[j] / num_samples;
        }
    }
}

// kernel launch
void gradientDescent(float *X, float *y, float *theta, int num_samples, int num_features, float lr, int iterations) {
    float *d_X, *d_y, *d_theta, *d_grad;
    size_t size_X = num_samples * num_features * sizeof(float);
    size_t size_y = num_samples * sizeof(float);
    size_t size_theta = num_features * sizeof(float);

    cudaMalloc((void**)&d_X, size_X);
    cudaMalloc((void**)&d_y, size_y);
    cudaMalloc((void**)&d_theta, size_theta);
    cudaMalloc((void**)&d_grad, sizeof(float));
    cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size_y, cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta, size_theta, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int iter = 0; iter < iterations; iter++) {
        cudaMemset(d_grad, 0, sizeof(float));
        gradientDescentKernel<<<gridDim, blockDim>>>(d_X, d_y, d_theta, d_grad, num_samples, num_features, lr);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(theta, d_theta, size_theta, cudaMemcpyDeviceToHost);

    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_theta);
    cudaFree(d_grad);
}

void randomInit(float *A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
    }
}

int main() {
    int num_samples = 1000;
    int num_features = 10;
    int iterations = 1000;
    float lr = 0.01f;

    float *X = new float[num_samples * num_features];
    float *y = new float[num_samples];
    float *theta = new float[num_features];

    randomInit(X, num_samples * num_features);
    randomInit(y, num_samples);
    randomInit(theta, num_features);

    std::cout << "Running Optimized GPU Gradient Descent..." << std::endl;
    gradientDescent(X, y, theta, num_samples, num_features, lr, iterations);

    delete[] X;
    delete[] y;
    delete[] theta;

    return 0;
}
