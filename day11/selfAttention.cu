#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16  // Tiling for shared memory
#define EPSILON 1e-6  
// optimized self-attention kernel with shared memory
__global__ void selfAttentionKernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    int N, // seq length
    int d  // embedding dim
) {
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_V[TILE_SIZE][TILE_SIZE];

    const int i = blockIdx.x * TILE_SIZE + threadIdx.x;  // sequence position
    const int k = blockIdx.y * TILE_SIZE + threadIdx.y;  // embedding dim

    if (i >= N || k >= d) return;

    float sum_exp = 0.0f;
    float acc = 0.0f;

    // iterate over tiled chunks of K and V
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tile_idx = t * TILE_SIZE + threadIdx.y;
        if (tile_idx < N) {
            tile_Q[threadIdx.x][threadIdx.y] = Q[i * d + tile_idx];
            tile_K[threadIdx.x][threadIdx.y] = K[tile_idx * d + k];
            tile_V[threadIdx.x][threadIdx.y] = V[tile_idx * d + k];
        } else {
            tile_Q[threadIdx.x][threadIdx.y] = 0.0f;
            tile_K[threadIdx.x][threadIdx.y] = 0.0f;
            tile_V[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();

        // compute scaled dot-product attention
        for (int j = 0; j < TILE_SIZE; j++) {
            if (t * TILE_SIZE + j < N) {
                float score = 0.0f;
                for (int l = 0; l < d; l++) {
                    score += tile_Q[threadIdx.x][l] * tile_K[j][l];
                }
                score /= sqrtf(d);

                float exp_score = expf(score);
                sum_exp += exp_score;
                acc += exp_score * tile_V[j][threadIdx.y];
            }
        }
        __syncthreads();
    }

    // normalize with softmax
    O[i * d + k] = acc / (sum_exp + EPSILON);
}

// host function for self-attention
void selfAttention(const float *Q, const float *K, const float *V, float *O, int N, int d) {
    float *dQ, *dK, *dV, *dO;
    size_t sizeMatrix = N * d * sizeof(float);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (d + TILE_SIZE - 1) / TILE_SIZE);

    cudaMalloc((void **)&dQ, sizeMatrix);
    cudaMalloc((void **)&dK, sizeMatrix);
    cudaMalloc((void **)&dV, sizeMatrix);
    cudaMalloc((void **)&dO, sizeMatrix);
    
    cudaMemcpy(dQ, Q, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, sizeMatrix, cudaMemcpyHostToDevice);

    // launch kernel
    selfAttentionKernel<<<gridDim, blockDim>>>(dQ, dK, dV, dO, N, d);
    cudaDeviceSynchronize();

    cudaMemcpy(O, dO, sizeMatrix, cudaMemcpyDeviceToHost);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
}

// initialize random values
void randominit(float *A, const int N) {
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// main function
int main() {
    int N = 32;  // Sequence length
    int d = 64;  // Embedding dimension

    float *Q = new float[N * d];
    float *K = new float[N * d];
    float *V = new float[N * d];
    float *O = new float[N * d];

    randominit(Q, N * d);
    randominit(K, N * d);
    randominit(V, N * d);

    selfAttention(Q, K, V, O, N, d);

    std::cout << "First 4x4 block of output O:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << O[i * d + j] << " ";
        }
        std::cout << std::endl;
    }


    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] O;

    return 0;
}
