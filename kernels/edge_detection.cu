#include "cuda_kernels.h"

// ---------------------------------------------------------------------------
// Sobel Edge Detection Kernel
// ---------------------------------------------------------------------------

__global__ void sobelEdgeKernel(const uint8_t* input, uint8_t* output,
                                int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Skip border pixels
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    // Sobel X kernel:  [-1, 0, 1]    Sobel Y kernel:  [-1, -2, -1]
    //                  [-2, 0, 2]                     [ 0,  0,  0]
    //                  [-1, 0, 1]                     [ 1,  2,  1]

    float gx = -1.0f * input[(y-1) * width + (x-1)] +  1.0f * input[(y-1) * width + (x+1)]
             + -2.0f * input[(y)   * width + (x-1)] +  2.0f * input[(y)   * width + (x+1)]
             + -1.0f * input[(y+1) * width + (x-1)] +  1.0f * input[(y+1) * width + (x+1)];

    float gy = -1.0f * input[(y-1) * width + (x-1)] + -2.0f * input[(y-1) * width + x] + -1.0f * input[(y-1) * width + (x+1)]
             +  1.0f * input[(y+1) * width + (x-1)] +  2.0f * input[(y+1) * width + x] +  1.0f * input[(y+1) * width + (x+1)];

    float magnitude = sqrtf(gx * gx + gy * gy);

    // Normalize magnitude (max possible is ~1442 for 8-bit input)
    float normalized = magnitude / 1442.0f;

    output[y * width + x] = (normalized > threshold)
        ? static_cast<uint8_t>(min(normalized * 255.0f, 255.0f))
        : 0;
}

void launchSobelEdgeDetection(const uint8_t* d_input, uint8_t* d_output,
                              int width, int height, float threshold) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    sobelEdgeKernel<<<grid, block>>>(d_input, d_output, width, height, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Canny Edge Detection (simplified) Kernel
// ---------------------------------------------------------------------------

// Step 1: Compute gradient magnitude and direction
__global__ void cannyGradientKernel(const uint8_t* input,
                                     float* magnitude, float* direction,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        magnitude[y * width + x] = 0.0f;
        direction[y * width + x] = 0.0f;
        return;
    }

    float gx = -1.0f * input[(y-1) * width + (x-1)] +  1.0f * input[(y-1) * width + (x+1)]
             + -2.0f * input[(y)   * width + (x-1)] +  2.0f * input[(y)   * width + (x+1)]
             + -1.0f * input[(y+1) * width + (x-1)] +  1.0f * input[(y+1) * width + (x+1)];

    float gy = -1.0f * input[(y-1) * width + (x-1)] + -2.0f * input[(y-1) * width + x] + -1.0f * input[(y-1) * width + (x+1)]
             +  1.0f * input[(y+1) * width + (x-1)] +  2.0f * input[(y+1) * width + x] +  1.0f * input[(y+1) * width + (x+1)];

    magnitude[y * width + x] = sqrtf(gx * gx + gy * gy);
    direction[y * width + x] = atan2f(gy, gx);
}

// Step 2: Non-maximum suppression
__global__ void cannyNmsKernel(const float* magnitude, const float* direction,
                                uint8_t* output, int width, int height,
                                float lowThreshold, float highThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 1 || x >= width - 2 || y <= 1 || y >= height - 2) {
        if (x < width && y < height) output[y * width + x] = 0;
        return;
    }

    float mag = magnitude[y * width + x];
    float angle = direction[y * width + x];

    // Quantize angle to 4 directions
    if (angle < 0) angle += 3.14159265f;
    float deg = angle * 180.0f / 3.14159265f;

    float mag1 = 0.0f, mag2 = 0.0f;

    if ((deg >= 0 && deg < 22.5f) || (deg >= 157.5f && deg <= 180.0f)) {
        mag1 = magnitude[y * width + (x + 1)];
        mag2 = magnitude[y * width + (x - 1)];
    } else if (deg >= 22.5f && deg < 67.5f) {
        mag1 = magnitude[(y - 1) * width + (x + 1)];
        mag2 = magnitude[(y + 1) * width + (x - 1)];
    } else if (deg >= 67.5f && deg < 112.5f) {
        mag1 = magnitude[(y - 1) * width + x];
        mag2 = magnitude[(y + 1) * width + x];
    } else {
        mag1 = magnitude[(y - 1) * width + (x - 1)];
        mag2 = magnitude[(y + 1) * width + (x + 1)];
    }

    // Non-maximum suppression
    if (mag >= mag1 && mag >= mag2) {
        float normalizedMag = mag / 1442.0f;
        if (normalizedMag >= highThreshold) {
            output[y * width + x] = 255;  // Strong edge
        } else if (normalizedMag >= lowThreshold) {
            output[y * width + x] = 128;  // Weak edge
        } else {
            output[y * width + x] = 0;
        }
    } else {
        output[y * width + x] = 0;
    }
}

void launchCannyEdgeDetection(const uint8_t* d_input, uint8_t* d_output,
                              int width, int height,
                              float lowThreshold, float highThreshold) {
    float* d_magnitude = nullptr;
    float* d_direction = nullptr;

    size_t pixelCount = static_cast<size_t>(width) * height;
    CUDA_CHECK(cudaMalloc(&d_magnitude, pixelCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_direction, pixelCount * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Step 1: Compute gradients
    cannyGradientKernel<<<grid, block>>>(d_input, d_magnitude, d_direction,
                                         width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Non-maximum suppression with double thresholding
    cannyNmsKernel<<<grid, block>>>(d_magnitude, d_direction, d_output,
                                    width, height, lowThreshold, highThreshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_magnitude));
    CUDA_CHECK(cudaFree(d_direction));
}
