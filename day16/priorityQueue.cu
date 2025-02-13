#include <iostream>
#include <cuda_runtime.h>
#include <limits>

#define BLOCK_SIZE 256
#define HEAP_SIZE 32  // Maximum elements in the priority queue

__device__ void heapifyUp(float* heap, int* indices, int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap[idx] < heap[parent]) {
            // Swap elements
            float temp = heap[idx];
            heap[idx] = heap[parent];
            heap[parent] = temp;
            
            int tempIdx = indices[idx];
            indices[idx] = indices[parent];
            indices[parent] = tempIdx;

            idx = parent;
        } else {
            break;
        }
    }
}

__device__ void heapifyDown(float* heap, int* indices, int size, int idx) {
    while (2 * idx + 1 < size) {
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;
        int smallest = idx;

        if (left < size && heap[left] < heap[smallest]) smallest = left;
        if (right < size && heap[right] < heap[smallest]) smallest = right;

        if (smallest != idx) {
            float temp = heap[idx];
            heap[idx] = heap[smallest];
            heap[smallest] = temp;

            int tempIdx = indices[idx];
            indices[idx] = indices[smallest];
            indices[smallest] = tempIdx;

            idx = smallest;
        } else {
            break;
        }
    }
}

__global__ void parallelPriorityQueue(float* input, float* output, int* out_indices, int num_elements, int k) {
    __shared__ float heap[HEAP_SIZE];  
    __shared__ int indices[HEAP_SIZE];  
    __shared__ int heapSize;  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x == 0) heapSize = 0;
    __syncthreads();

    if (tid < num_elements) {
        float val = input[tid];

        int insertPos;
        if (heapSize < k) {
            insertPos = atomicAdd(&heapSize, 1);
        } else {
            insertPos = k - 1;
            if (val >= heap[insertPos]) return;
        }

        heap[insertPos] = val;
        indices[insertPos] = tid;
        __syncthreads();

        if (insertPos < k) heapifyUp(heap, indices, insertPos);
    }
    __syncthreads();

    if (threadIdx.x == 0 && heapSize > 1) {
        for (int i = heapSize - 1; i >= 0; i--) {
            float minVal = heap[0];
            int minIdx = indices[0];

            heap[0] = heap[i];
            indices[0] = indices[i];

            heapifyDown(heap, indices, i, 0);

            output[i] = minVal;
            out_indices[i] = minIdx;
        }
    }
}

void runPriorityQueue(float* input, float* output, int* out_indices, int num_elements, int k) {
    float* d_input, * d_output;
    int* d_indices;
    
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_output, k * sizeof(float));
    cudaMalloc(&d_indices, k * sizeof(int));

    cudaMemcpy(d_input, input, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    parallelPriorityQueue<<<gridDim, blockDim>>>(d_input, d_output, d_indices, num_elements, k);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_indices, d_indices, k * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
}

void randomInit(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
}

int main() {
    int num_elements = 1024;
    int k = 10;
    
    float* input = new float[num_elements];
    float* output = new float[k];
    int* out_indices = new int[k];

    randomInit(input, num_elements);
    
    runPriorityQueue(input, output, out_indices, num_elements, k);

    std::cout << "Top " << k << " elements:" << std::endl;
    for (int i = 0; i < k; i++) {
        std::cout << "Value: " << output[i] << " (Index: " << out_indices[i] << ")" << std::endl;
    }

    delete[] input;
    delete[] output;
    delete[] out_indices;

    return 0;
}
