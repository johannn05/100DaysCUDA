# 100 Days of CUDA

This repository serves to track my progress in learning parallel programming through writing CUDA kernels. I use the PMPP book to learn concepts that I apply through my kernels here.

Mentor: https://github.com/hkproj/


### Daily Updates
| Day   | File Description                                                                                                                                                                                                                      |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | **vectorAdd.cu**:  GPU vector addition: memory allocation, host-to-device data transfer, kernel launch.<br>**matrixAdd.cu**: GPU matrix addition: 2D grid/block indexing.
| 2 | **matrixMul.cu**:  matrix multiplication: element-wise computation using 2D thread blocks
| 3 | **rgb2Grey.cu**: image processing: parallel pixel-wise RGB to grayscale conversion using the luminance formula.<br>**CUDA_RGB_to_Grayscale.ipynb**: RGB conversion in Python using CUDA kernel integrated with PyTorch’s C++/CUDA extensions.
| 4 | **tiledMatmul.cu**: tiled matrix multiplication: shared memory, tiling, thread synchronization to optimize memory access and speeds
| 5 | **prefixSum.cu**: Prefix sum: shared memory, parallel prefix scan, thread synchronization
| 6 | **Softmax.cu**: Naive softmax: each thread computes softmax for one element using global memory only.<br>**layerNorm.cu**: each thread computes layer norm for one element using global memory, recomputing mean/variance redundantly.


To test my kernels, run them at https://leetgpu.com
