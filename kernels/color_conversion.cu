#include "cuda_kernels.h"

// ---------------------------------------------------------------------------
// RGB to Grayscale Kernel
// ---------------------------------------------------------------------------

__global__ void rgbToGrayscaleKernel(const uint8_t* input, uint8_t* output,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint8_t r = input[idx * 3 + 0];
    uint8_t g = input[idx * 3 + 1];
    uint8_t b = input[idx * 3 + 2];

    // ITU-R BT.601 luma coefficients
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    output[idx] = static_cast<uint8_t>(min(max(gray, 0.0f), 255.0f));
}

void launchRgbToGrayscale(const uint8_t* d_input, uint8_t* d_output,
                          int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rgbToGrayscaleKernel<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// RGB to HSV Kernel
// ---------------------------------------------------------------------------

__global__ void rgbToHsvKernel(const uint8_t* input, float* output,
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float r = input[idx * 3 + 0] / 255.0f;
    float g = input[idx * 3 + 1] / 255.0f;
    float b = input[idx * 3 + 2] / 255.0f;

    float maxVal = fmaxf(r, fmaxf(g, b));
    float minVal = fminf(r, fminf(g, b));
    float diff = maxVal - minVal;

    // Hue
    float h = 0.0f;
    if (diff > 0.0f) {
        if (maxVal == r) {
            h = 60.0f * fmodf((g - b) / diff, 6.0f);
        } else if (maxVal == g) {
            h = 60.0f * ((b - r) / diff + 2.0f);
        } else {
            h = 60.0f * ((r - g) / diff + 4.0f);
        }
    }
    if (h < 0.0f) h += 360.0f;

    // Saturation
    float s = (maxVal > 0.0f) ? (diff / maxVal) : 0.0f;

    // Value
    float v = maxVal;

    output[idx * 3 + 0] = h;
    output[idx * 3 + 1] = s;
    output[idx * 3 + 2] = v;
}

void launchRgbToHsv(const uint8_t* d_input, float* d_output,
                    int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rgbToHsvKernel<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Brightness and Contrast Adjustment Kernel
// ---------------------------------------------------------------------------

__global__ void brightnessContrastKernel(const uint8_t* input, uint8_t* output,
                                         int width, int height, int channels,
                                         float brightness, float contrast) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        int idx = (y * width + x) * channels + c;
        float val = input[idx];
        // Apply contrast around midpoint (128), then add brightness
        val = contrast * (val - 128.0f) + 128.0f + brightness;
        output[idx] = static_cast<uint8_t>(min(max(val, 0.0f), 255.0f));
    }
}

void launchBrightnessContrast(const uint8_t* d_input, uint8_t* d_output,
                              int width, int height, int channels,
                              float brightness, float contrast) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    brightnessContrastKernel<<<grid, block>>>(d_input, d_output, width, height,
                                              channels, brightness, contrast);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
