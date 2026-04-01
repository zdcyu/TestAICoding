#include "image_processor.h"

#include <fstream>
#include <sstream>
#include <algorithm>

ImageProcessor::ImageProcessor()
    : deviceId_(0), initialized_(false), deviceProps_{} {}

ImageProcessor::~ImageProcessor() {
    if (initialized_) {
        cudaDeviceReset();
    }
}

bool ImageProcessor::initialize(int deviceId) {
    deviceId_ = deviceId;

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceId_ >= deviceCount) {
        std::cerr << "Error: CUDA device " << deviceId_ << " not found. "
                  << deviceCount << " device(s) available." << std::endl;
        return false;
    }

    CUDA_CHECK(cudaSetDevice(deviceId_));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps_, deviceId_));

    initialized_ = true;
    return true;
}

void ImageProcessor::printDeviceInfo() const {
    if (!initialized_) {
        std::cerr << "ImageProcessor not initialized." << std::endl;
        return;
    }

    std::cout << "=== CUDA Device Info ===" << std::endl;
    std::cout << "Device:              " << deviceProps_.name << std::endl;
    std::cout << "Compute Capability:  " << deviceProps_.major << "."
              << deviceProps_.minor << std::endl;
    std::cout << "Total Global Memory: "
              << deviceProps_.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Memory/Block: "
              << deviceProps_.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max Threads/Block:   " << deviceProps_.maxThreadsPerBlock << std::endl;
    std::cout << "Warp Size:           " << deviceProps_.warpSize << std::endl;
    std::cout << "Multiprocessors:     " << deviceProps_.multiProcessorCount << std::endl;
    std::cout << "========================" << std::endl;
}

// ---------------------------------------------------------------------------
// GPU memory helpers
// ---------------------------------------------------------------------------

uint8_t* ImageProcessor::allocateDevice(size_t size) {
    uint8_t* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, size));
    return d_ptr;
}

void ImageProcessor::freeDevice(uint8_t* d_ptr) {
    if (d_ptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
}

void ImageProcessor::copyToDevice(uint8_t* d_dst, const uint8_t* h_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
}

void ImageProcessor::copyToHost(uint8_t* h_dst, const uint8_t* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Image filter operations
// ---------------------------------------------------------------------------

Image ImageProcessor::gaussianBlur(const Image& input, int kernelRadius, float sigma) {
    Image output(input.width, input.height, input.channels);

    uint8_t* d_input  = allocateDevice(input.sizeInBytes());
    uint8_t* d_output = allocateDevice(output.sizeInBytes());

    copyToDevice(d_input, input.data.data(), input.sizeInBytes());

    GpuTimer timer;
    timer.start();
    launchGaussianBlur(d_input, d_output, input.width, input.height,
                       input.channels, kernelRadius, sigma);
    timer.stop();
    std::cout << "Gaussian Blur: " << timer.elapsedMs() << " ms" << std::endl;

    copyToHost(output.data.data(), d_output, output.sizeInBytes());

    freeDevice(d_input);
    freeDevice(d_output);

    return output;
}

Image ImageProcessor::boxBlur(const Image& input, int kernelRadius) {
    Image output(input.width, input.height, input.channels);

    uint8_t* d_input  = allocateDevice(input.sizeInBytes());
    uint8_t* d_output = allocateDevice(output.sizeInBytes());

    copyToDevice(d_input, input.data.data(), input.sizeInBytes());

    GpuTimer timer;
    timer.start();
    launchBoxBlur(d_input, d_output, input.width, input.height,
                  input.channels, kernelRadius);
    timer.stop();
    std::cout << "Box Blur: " << timer.elapsedMs() << " ms" << std::endl;

    copyToHost(output.data.data(), d_output, output.sizeInBytes());

    freeDevice(d_input);
    freeDevice(d_output);

    return output;
}

Image ImageProcessor::sharpen(const Image& input, float strength) {
    Image output(input.width, input.height, input.channels);

    uint8_t* d_input  = allocateDevice(input.sizeInBytes());
    uint8_t* d_output = allocateDevice(output.sizeInBytes());

    copyToDevice(d_input, input.data.data(), input.sizeInBytes());

    GpuTimer timer;
    timer.start();
    launchSharpen(d_input, d_output, input.width, input.height,
                  input.channels, strength);
    timer.stop();
    std::cout << "Sharpen: " << timer.elapsedMs() << " ms" << std::endl;

    copyToHost(output.data.data(), d_output, output.sizeInBytes());

    freeDevice(d_input);
    freeDevice(d_output);

    return output;
}

// ---------------------------------------------------------------------------
// Color conversion operations
// ---------------------------------------------------------------------------

Image ImageProcessor::rgbToGrayscale(const Image& input) {
    if (input.channels != 3) {
        std::cerr << "Error: rgbToGrayscale expects a 3-channel image." << std::endl;
        return {};
    }

    Image output(input.width, input.height, 1);

    uint8_t* d_input  = allocateDevice(input.sizeInBytes());
    uint8_t* d_output = allocateDevice(output.sizeInBytes());

    copyToDevice(d_input, input.data.data(), input.sizeInBytes());

    GpuTimer timer;
    timer.start();
    launchRgbToGrayscale(d_input, d_output, input.width, input.height);
    timer.stop();
    std::cout << "RGB to Grayscale: " << timer.elapsedMs() << " ms" << std::endl;

    copyToHost(output.data.data(), d_output, output.sizeInBytes());

    freeDevice(d_input);
    freeDevice(d_output);

    return output;
}

Image ImageProcessor::brightnessContrast(const Image& input,
                                          float brightness, float contrast) {
    Image output(input.width, input.height, input.channels);

    uint8_t* d_input  = allocateDevice(input.sizeInBytes());
    uint8_t* d_output = allocateDevice(output.sizeInBytes());

    copyToDevice(d_input, input.data.data(), input.sizeInBytes());

    GpuTimer timer;
    timer.start();
    launchBrightnessContrast(d_input, d_output, input.width, input.height,
                             input.channels, brightness, contrast);
    timer.stop();
    std::cout << "Brightness/Contrast: " << timer.elapsedMs() << " ms" << std::endl;

    copyToHost(output.data.data(), d_output, output.sizeInBytes());

    freeDevice(d_input);
    freeDevice(d_output);

    return output;
}

// ---------------------------------------------------------------------------
// Edge detection operations
// ---------------------------------------------------------------------------

Image ImageProcessor::sobelEdgeDetection(const Image& input, float threshold) {
    // Convert to grayscale first if needed
    Image grayInput = (input.channels == 1) ? input : rgbToGrayscale(input);

    Image output(grayInput.width, grayInput.height, 1);

    uint8_t* d_input  = allocateDevice(grayInput.sizeInBytes());
    uint8_t* d_output = allocateDevice(output.sizeInBytes());

    copyToDevice(d_input, grayInput.data.data(), grayInput.sizeInBytes());

    GpuTimer timer;
    timer.start();
    launchSobelEdgeDetection(d_input, d_output, grayInput.width,
                             grayInput.height, threshold);
    timer.stop();
    std::cout << "Sobel Edge Detection: " << timer.elapsedMs() << " ms" << std::endl;

    copyToHost(output.data.data(), d_output, output.sizeInBytes());

    freeDevice(d_input);
    freeDevice(d_output);

    return output;
}

Image ImageProcessor::cannyEdgeDetection(const Image& input,
                                          float lowThreshold,
                                          float highThreshold) {
    Image grayInput = (input.channels == 1) ? input : rgbToGrayscale(input);

    Image output(grayInput.width, grayInput.height, 1);

    uint8_t* d_input  = allocateDevice(grayInput.sizeInBytes());
    uint8_t* d_output = allocateDevice(output.sizeInBytes());

    copyToDevice(d_input, grayInput.data.data(), grayInput.sizeInBytes());

    GpuTimer timer;
    timer.start();
    launchCannyEdgeDetection(d_input, d_output, grayInput.width,
                             grayInput.height, lowThreshold, highThreshold);
    timer.stop();
    std::cout << "Canny Edge Detection: " << timer.elapsedMs() << " ms" << std::endl;

    copyToHost(output.data.data(), d_output, output.sizeInBytes());

    freeDevice(d_input);
    freeDevice(d_output);

    return output;
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

Image ImageProcessor::generateTestImage(int width, int height) {
    Image img(width, height, 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            img.data[idx + 0] = static_cast<uint8_t>((x * 255) / width);   // R gradient
            img.data[idx + 1] = static_cast<uint8_t>((y * 255) / height);  // G gradient
            img.data[idx + 2] = static_cast<uint8_t>(128);                 // B constant
        }
    }

    return img;
}

bool ImageProcessor::saveImagePPM(const Image& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return false;
    }

    if (img.channels == 1) {
        // PGM format for grayscale
        file << "P5\n" << img.width << " " << img.height << "\n255\n";
    } else {
        // PPM format for RGB
        file << "P6\n" << img.width << " " << img.height << "\n255\n";
    }

    file.write(reinterpret_cast<const char*>(img.data.data()), img.sizeInBytes());
    file.close();

    std::cout << "Saved image: " << filename << " (" << img.width << "x"
              << img.height << ", " << img.channels << " channels)" << std::endl;
    return true;
}

Image ImageProcessor::loadImagePPM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for reading." << std::endl;
        return {};
    }

    std::string magic;
    int width, height, maxVal;
    file >> magic >> width >> height >> maxVal;
    file.get(); // consume newline

    int channels = (magic == "P6") ? 3 : 1;
    Image img(width, height, channels);
    file.read(reinterpret_cast<char*>(img.data.data()), img.sizeInBytes());
    file.close();

    std::cout << "Loaded image: " << filename << " (" << width << "x"
              << height << ", " << channels << " channels)" << std::endl;
    return img;
}
