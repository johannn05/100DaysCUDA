#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <stb_image.h>
#include <stb_image_write.h>

// RGB to greyscale kernel
__global__ void rgb_to_grayscale_kernel(unsigned char* rgb, unsigned char* grayscale, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        unsigned char r = rgb[idx];
        unsigned char g = rgb[idx + total_pixels];
        unsigned char b = rgb[idx + 2 * total_pixels];
        
        // greyscale formula
        grayscale[idx] = static_cast<unsigned char>(0.2989f * r + 0.5870f * g + 0.1140f * b);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    // load img using stb_image
    int width, height, channels;
    unsigned char* host_rgb = stbi_load(input_path, &width, &height, &channels, 3); // force 3 channels (RGB)

    if (!host_rgb) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return 1;
    }

    size_t rgb_size = width * height * 3 * sizeof(unsigned char);
    size_t grayscale_size = width * height * sizeof(unsigned char);

    // allocate memory on device
    unsigned char *dev_rgb, *dev_grayscale;
    cudaMalloc(&dev_rgb, rgb_size);
    cudaMalloc(&dev_grayscale, grayscale_size);

    cudaMemcpy(dev_rgb, host_rgb, rgb_size, cudaMemcpyHostToDevice);

    // define block and grid sizes
    int threads_per_block = 256;
    int total_pixels = width * height;
    int num_blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    // kernel launch
    rgb_to_grayscale_kernel<<<num_blocks, threads_per_block>>>(dev_rgb, dev_grayscale, width, height);

    cudaDeviceSynchronize();

    // copy result back to the host
    unsigned char* host_grayscale = (unsigned char*)malloc(grayscale_size);
    cudaMemcpy(host_grayscale, dev_grayscale, grayscale_size, cudaMemcpyDeviceToHost);

    // save grayscale image using stb_image_write
    if (!stbi_write_png(output_path, width, height, 1, host_grayscale, width)) {
        std::cerr << "Failed to save image: " << output_path << std::endl;
        return 1;
    }

    std::cout << "Image converted to grayscale and saved to " << output_path << std::endl;

    // free memory
    stbi_image_free(host_rgb);
    free(host_grayscale);
    cudaFree(dev_rgb);
    cudaFree(dev_grayscale);

    return 0;
}
