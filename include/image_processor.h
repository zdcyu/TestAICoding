#pragma once

#include "common.h"
#include "cuda_kernels.h"

// High-level image processing class that wraps CUDA kernel calls
class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    // Initialize CUDA device
    bool initialize(int deviceId = 0);

    // Print CUDA device information
    void printDeviceInfo() const;

    // Image filter operations
    Image gaussianBlur(const Image& input, int kernelRadius = 2, float sigma = 1.0f);
    Image boxBlur(const Image& input, int kernelRadius = 2);
    Image sharpen(const Image& input, float strength = 1.0f);

    // Color conversion operations
    Image rgbToGrayscale(const Image& input);
    Image brightnessContrast(const Image& input, float brightness = 0.0f, float contrast = 1.0f);

    // Edge detection operations
    Image sobelEdgeDetection(const Image& input, float threshold = 0.1f);
    Image cannyEdgeDetection(const Image& input, float lowThreshold = 0.05f, float highThreshold = 0.15f);

    // Generate a test image (gradient pattern)
    static Image generateTestImage(int width, int height);

    // Save image as PPM file (simple format, no external dependencies)
    static bool saveImagePPM(const Image& img, const std::string& filename);

    // Load image from PPM file
    static Image loadImagePPM(const std::string& filename);

private:
    int deviceId_;
    bool initialized_;
    cudaDeviceProp deviceProps_;

    // Allocate and free GPU memory helpers
    uint8_t* allocateDevice(size_t size);
    void freeDevice(uint8_t* d_ptr);

    // Transfer data between host and device
    void copyToDevice(uint8_t* d_dst, const uint8_t* h_src, size_t size);
    void copyToHost(uint8_t* h_dst, const uint8_t* d_src, size_t size);
};
