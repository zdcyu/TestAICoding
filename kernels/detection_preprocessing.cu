#include "cuda_kernels.h"

// ---------------------------------------------------------------------------
// Image Resize Kernel (Bilinear Interpolation) for detection preprocessing
// ---------------------------------------------------------------------------

__global__ void resizeKernel(const uint8_t* input, uint8_t* output,
                             int srcWidth, int srcHeight,
                             int dstWidth, int dstHeight,
                             int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight) return;

    float scaleX = static_cast<float>(srcWidth) / dstWidth;
    float scaleY = static_cast<float>(srcHeight) / dstHeight;

    float srcX = (x + 0.5f) * scaleX - 0.5f;
    float srcY = (y + 0.5f) * scaleY - 0.5f;

    int x0 = static_cast<int>(floorf(srcX));
    int y0 = static_cast<int>(floorf(srcY));
    int x1 = min(x0 + 1, srcWidth - 1);
    int y1 = min(y0 + 1, srcHeight - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float fx = srcX - x0;
    float fy = srcY - y0;

    for (int c = 0; c < channels; ++c) {
        float v00 = input[(y0 * srcWidth + x0) * channels + c];
        float v01 = input[(y0 * srcWidth + x1) * channels + c];
        float v10 = input[(y1 * srcWidth + x0) * channels + c];
        float v11 = input[(y1 * srcWidth + x1) * channels + c];

        float val = (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) +
                    fy * ((1.0f - fx) * v10 + fx * v11);

        output[(y * dstWidth + x) * channels + c] =
            static_cast<uint8_t>(min(max(val, 0.0f), 255.0f));
    }
}

// ---------------------------------------------------------------------------
// BGR to RGB + Normalize Kernel (for YOLO input preprocessing)
// Converts uint8 BGR image to float32 RGB normalized [0, 1]
// ---------------------------------------------------------------------------

__global__ void bgrToRgbNormalizeKernel(const uint8_t* input, float* output,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;

    // BGR -> RGB and normalize to [0, 1]
    uint8_t b = input[pixelIdx * 3 + 0];
    uint8_t g = input[pixelIdx * 3 + 1];
    uint8_t r = input[pixelIdx * 3 + 2];

    // Output in CHW format (Channel, Height, Width) as expected by YOLO
    int planeSize = width * height;
    output[0 * planeSize + pixelIdx] = r / 255.0f;  // R channel
    output[1 * planeSize + pixelIdx] = g / 255.0f;  // G channel
    output[2 * planeSize + pixelIdx] = b / 255.0f;  // B channel
}

// ---------------------------------------------------------------------------
// Letterbox Resize Kernel
// Resizes image while maintaining aspect ratio and pads with gray (114)
// ---------------------------------------------------------------------------

__global__ void letterboxKernel(const uint8_t* input, uint8_t* output,
                                int srcWidth, int srcHeight,
                                int dstWidth, int dstHeight,
                                int channels,
                                float scale, int padX, int padY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight) return;

    int outIdx = (y * dstWidth + x) * channels;

    // Check if pixel is in the padded region
    int srcPixelX = static_cast<int>((x - padX) / scale);
    int srcPixelY = static_cast<int>((y - padY) / scale);

    if (srcPixelX < 0 || srcPixelX >= srcWidth ||
        srcPixelY < 0 || srcPixelY >= srcHeight) {
        // Padding region - fill with gray (114)
        for (int c = 0; c < channels; ++c) {
            output[outIdx + c] = 114;
        }
    } else {
        // Copy source pixel
        int srcIdx = (srcPixelY * srcWidth + srcPixelX) * channels;
        for (int c = 0; c < channels; ++c) {
            output[outIdx + c] = input[srcIdx + c];
        }
    }
}
