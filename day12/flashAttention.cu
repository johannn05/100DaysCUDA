#include <iostream>
#include <cuda_runtime.h>
#include <math_constants.h>

// constants for Flash Attention
const int M = 32;  // seq length
const int d_embed = 32;  // Embedding dim
const int Bc = (M / (4 * d_embed)) > 0 ? (M / (4 * d_embed)) : 1;  // nonzero Bc
const int Br = (Bc < d_embed) ? Bc : d_embed;  
const int Tr = (M + Br - 1) / Br;  // compute Tr at runtime
const int Tc = (M + Bc - 1) / Bc;  // compute Tc at runtime

// optimized flash attention kernel
__global__ void flashAttentionKernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    float *__restrict__ m,
    float *__restrict__ l)
{
    // 
    __shared__ float QBlock[Br][d_embed];  
    __shared__ float KBlock[Bc][d_embed];  
    __shared__ float VBlock[Bc][d_embed];  
    __shared__ float OBlock[Br][d_embed];  
    __shared__ float lBlock[Br];       
    __shared__ float mBlock[Br];       

    const int i = blockIdx.x * Br + threadIdx.x;  // row index in Q
    const int j = blockIdx.y * Bc + threadIdx.y;  // column index in K/V

    if (i >= M || j >= d_embed) return;

    float rowMax = -CUDART_INF_F;
    float sumExp = 0.0f;

    // load Q, K, V blocks into shared memory
    for (int p = 0; p < Br; ++p) {
        QBlock[p][threadIdx.y] = Q[(blockIdx.x * Br + p) * d_embed + threadIdx.y];
    }
    for (int p = 0; p < Bc; ++p) {
        KBlock[p][threadIdx.x] = K[(blockIdx.y * Bc + p) * d_embed + threadIdx.x];
        VBlock[p][threadIdx.x] = V[(blockIdx.y * Bc + p) * d_embed + threadIdx.x];
    }

    __syncthreads();

    // compute scaled dot-product attention
    float S[Br][Bc];
    float P[Br][Bc];

    for (int p = 0; p < Bc; ++p) {
        float score = 0.0f;
        for (int k = 0; k < d_embed; ++k) {
            score += QBlock[threadIdx.x][k] * KBlock[p][k];
        }
        score /= sqrtf(d_embed);  

        S[threadIdx.x][p] = score;
        rowMax = fmaxf(rowMax, score);
    }

    // compute stable softmax scores
    for (int p = 0; p < Bc; ++p) {
        P[threadIdx.x][p] = expf(S[threadIdx.x][p] - rowMax);
        sumExp += P[threadIdx.x][p];
    }

    // store new row max and sum for normalization
    mBlock[threadIdx.x] = rowMax;
    lBlock[threadIdx.x] = sumExp;

    __syncthreads();

    // compute final attention output: O = P * V
    for (int k = 0; k < d_embed; ++k) {
        float sum = 0.0f;
        for (int p = 0; p < Bc; ++p) {
            sum += P[threadIdx.x][p] * VBlock[p][k];
        }
        OBlock[threadIdx.x][k] = sum / sumExp;
    }

    __syncthreads();

    // write back to global memory
    for (int k = 0; k < d_embed; ++k) {
        O[i * d_embed + k] = OBlock[threadIdx.x][k];
    }
}

// host function to launch flash attention kernel
void flashAttention(const float *Q, const float *K, const float *V, float *O, int N, int d) {
    float *dQ, *dK, *dV, *dO, *dM, *dL;
    size_t sizeMatrix = N * d * sizeof(float);
    size_t sizeVector = N * sizeof(float);

    cudaMalloc((void **)&dQ, sizeMatrix);
    cudaMalloc((void **)&dK, sizeMatrix);
    cudaMalloc((void **)&dV, sizeMatrix);
    cudaMalloc((void **)&dO, sizeMatrix);
    cudaMalloc((void **)&dM, sizeVector);
    cudaMalloc((void **)&dL, sizeVector);

    cudaMemcpy(dQ, Q, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, sizeMatrix, cudaMemcpyHostToDevice);

    // configure grid and block dimensions
    dim3 blockDim(Br, d_embed);
    dim3 gridDim(Tc, Tr);

    // launch optimized kernel
    flashAttentionKernel<<<gridDim, blockDim>>>(dQ, dK, dV, dO, dM, dL);
    cudaDeviceSynchronize();

    cudaMemcpy(O, dO, sizeMatrix, cudaMemcpyDeviceToHost);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    cudaFree(dM);
    cudaFree(dL);
}

// initialize random values
void randominit(float *A, const int N) {
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
    }
}

// main function
int main() {
    int N = 32;
    int d_embed = 64;

    float *Q = new float[N * d_embed];
    float *K = new float[N * d_embed];
    float *V = new float[N * d_embed];
    float *O = new float[N * d_embed];

    randominit(Q, N * d_embed);
    randominit(K, N * d_embed);
    randominit(V, N * d_embed);

    flashAttention(Q, K, V, O, N, d_embed);

    std::cout << "First 4x4 block of output O:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << O[i * d_embed + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] Q;
    delete[] K;
    delete[] V; 
    delete[] O;

    return 0;
}
