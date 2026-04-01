#pragma once

#include "common.h"

// ============================================================================
// Image Filter Kernels
// ============================================================================

// Apply Gaussian blur filter on GPU
void launchGaussianBlur(const uint8_t* d_input, uint8_t* d_output,
                        int width, int height, int channels,
                        int kernelRadius, float sigma);

// Apply box blur filter on GPU
void launchBoxBlur(const uint8_t* d_input, uint8_t* d_output,
                   int width, int height, int channels,
                   int kernelRadius);

// Apply sharpening filter on GPU
void launchSharpen(const uint8_t* d_input, uint8_t* d_output,
                   int width, int height, int channels,
                   float strength);

// ============================================================================
// Color Conversion Kernels
// ============================================================================

// Convert RGB image to grayscale on GPU
void launchRgbToGrayscale(const uint8_t* d_input, uint8_t* d_output,
                          int width, int height);

// Convert RGB image to HSV on GPU
void launchRgbToHsv(const uint8_t* d_input, float* d_output,
                    int width, int height);

// Adjust brightness and contrast on GPU
void launchBrightnessContrast(const uint8_t* d_input, uint8_t* d_output,
                              int width, int height, int channels,
                              float brightness, float contrast);

// ============================================================================
// Edge Detection Kernels
// ============================================================================

// Apply Sobel edge detection on GPU
void launchSobelEdgeDetection(const uint8_t* d_input, uint8_t* d_output,
                              int width, int height, float threshold);

// Apply Canny-style edge detection on GPU (simplified)
void launchCannyEdgeDetection(const uint8_t* d_input, uint8_t* d_output,
                              int width, int height,
                              float lowThreshold, float highThreshold);
