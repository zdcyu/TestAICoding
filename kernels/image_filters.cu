#include "cuda_kernels.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Gaussian Blur Kernel
// ---------------------------------------------------------------------------

__global__ void gaussianBlurKernel(const uint8_t* input, uint8_t* output,
                                   int width, int height, int channels,
                                   const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = kernelSize / 2;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                int sx = min(max(x + kx, 0), width - 1);
                int sy = min(max(y + ky, 0), height - 1);
                float weight = kernel[(ky + radius) * kernelSize + (kx + radius)];
                sum += weight * input[(sy * width + sx) * channels + c];
            }
        }
        output[(y * width + x) * channels + c] =
            static_cast<uint8_t>(min(max(sum, 0.0f), 255.0f));
    }
}

void launchGaussianBlur(const uint8_t* d_input, uint8_t* d_output,
                        int width, int height, int channels,
                        int kernelRadius, float sigma) {
    int kernelSize = 2 * kernelRadius + 1;
    int kernelElements = kernelSize * kernelSize;

    // Build Gaussian kernel on host
    std::vector<float> h_kernel(kernelElements);
    float sum = 0.0f;
    for (int y = -kernelRadius; y <= kernelRadius; ++y) {
        for (int x = -kernelRadius; x <= kernelRadius; ++x) {
            float val = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            h_kernel[(y + kernelRadius) * kernelSize + (x + kernelRadius)] = val;
            sum += val;
        }
    }
    for (auto& v : h_kernel) v /= sum;

    // Copy kernel to device
    float* d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelElements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(),
                          kernelElements * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    gaussianBlurKernel<<<grid, block>>>(d_input, d_output, width, height,
                                        channels, d_kernel, kernelSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_kernel));
}

// ---------------------------------------------------------------------------
// Box Blur Kernel
// ---------------------------------------------------------------------------

__global__ void boxBlurKernel(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int kernelSize = 2 * radius + 1;
    float invArea = 1.0f / (kernelSize * kernelSize);

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                int sx = min(max(x + kx, 0), width - 1);
                int sy = min(max(y + ky, 0), height - 1);
                sum += input[(sy * width + sx) * channels + c];
            }
        }
        output[(y * width + x) * channels + c] =
            static_cast<uint8_t>(min(max(sum * invArea, 0.0f), 255.0f));
    }
}

void launchBoxBlur(const uint8_t* d_input, uint8_t* d_output,
                   int width, int height, int channels, int kernelRadius) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    boxBlurKernel<<<grid, block>>>(d_input, d_output, width, height,
                                   channels, kernelRadius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Sharpen Kernel
// ---------------------------------------------------------------------------

__global__ void sharpenKernel(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Sharpening kernel: center = 1 + 4*strength, neighbors = -strength
    for (int c = 0; c < channels; ++c) {
        float center = input[(y * width + x) * channels + c];
        float top    = (y > 0)          ? input[((y-1) * width + x) * channels + c] : center;
        float bottom = (y < height - 1) ? input[((y+1) * width + x) * channels + c] : center;
        float left   = (x > 0)          ? input[(y * width + (x-1)) * channels + c] : center;
        float right  = (x < width - 1)  ? input[(y * width + (x+1)) * channels + c] : center;

        float val = center * (1.0f + 4.0f * strength) -
                    strength * (top + bottom + left + right);

        output[(y * width + x) * channels + c] =
            static_cast<uint8_t>(min(max(val, 0.0f), 255.0f));
    }
}

void launchSharpen(const uint8_t* d_input, uint8_t* d_output,
                   int width, int height, int channels, float strength) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    sharpenKernel<<<grid, block>>>(d_input, d_output, width, height,
                                   channels, strength);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
