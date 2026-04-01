#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// Image pixel structure (RGBA)
struct Pixel {
    uint8_t r, g, b, a;
};

// Image structure
struct Image {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;

    Image() : width(0), height(0), channels(0) {}
    Image(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c, 0) {}

    size_t sizeInBytes() const { return static_cast<size_t>(width) * height * channels; }
};

// Timer utility for benchmarking
class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { cudaEventRecord(start_, 0); }

    void stop() {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
    }

    float elapsedMs() {
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return elapsed;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
