# CUDA Computer Vision

A C++ project using NVIDIA CUDA for GPU-accelerated computer vision and image processing. Includes a Visual Studio solution (`.sln` / `.vcxproj`) for Windows development and CMake support for cross-platform builds.

## Features

- **Image Filters** - Gaussian blur, box blur, sharpening
- **Color Conversion** - RGB to grayscale, RGB to HSV, brightness/contrast adjustment
- **Edge Detection** - Sobel edge detection, Canny edge detection (simplified)
- **GPU Benchmarking** - Built-in timer for measuring kernel execution time
- **PPM/PGM I/O** - Simple image file format support (no external dependencies)

## Project Structure

```
CudaComputerVision/
├── CudaComputerVision.sln           # Visual Studio solution file
├── CudaComputerVision.vcxproj       # Visual Studio project file (CUDA 12.0)
├── CudaComputerVision.vcxproj.filters
├── CMakeLists.txt                   # CMake build (cross-platform)
├── include/
│   ├── common.h                     # Common types, CUDA error checking, timer
│   ├── cuda_kernels.h               # CUDA kernel launch function declarations
│   └── image_processor.h            # High-level ImageProcessor class
├── src/
│   ├── main.cpp                     # Entry point & demo pipeline
│   └── image_processor.cpp          # ImageProcessor implementation
├── kernels/
│   ├── image_filters.cu             # Gaussian blur, box blur, sharpen kernels
│   ├── color_conversion.cu          # RGB-Grayscale, RGB-HSV, brightness kernels
│   └── edge_detection.cu            # Sobel & Canny edge detection kernels
├── data/                            # Place input images here
└── build/                           # CMake build output (gitignored)
```

## Requirements

- **NVIDIA GPU** with Compute Capability 7.5+ (Turing / Ampere / Ada Lovelace)
- **CUDA Toolkit 12.0+** ([download](https://developer.nvidia.com/cuda-downloads))
- **C++17** compatible compiler

### Windows (Visual Studio)

- Visual Studio 2022 (v143 toolset)
- CUDA Toolkit 12.0 integrated with Visual Studio

### Linux / macOS (CMake)

- CMake 3.18+
- GCC 9+ or Clang 10+
- CUDA Toolkit 12.0+

## Building

### Visual Studio (Windows)

1. Open `CudaComputerVision.sln` in Visual Studio 2022
2. Ensure the CUDA 12.0 build customizations are installed
3. Select **Release | x64** configuration
4. Build the solution (**Ctrl+Shift+B**)

### CMake (Cross-Platform)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

To specify CUDA architectures:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
```

## Usage

```bash
# Run with auto-generated test image (1920x1080)
./bin/CudaComputerVision

# Run with custom test image size
./bin/CudaComputerVision --width 3840 --height 2160

# Run with an input PPM image
./bin/CudaComputerVision --input myimage.ppm --output results

# Select a specific GPU
./bin/CudaComputerVision --device 1
```

### Output

The program runs a full image processing pipeline and saves results as PPM/PGM files:

| Step | Output File | Description |
|------|-------------|-------------|
| 1 | `*_gaussian_blur.ppm` | Gaussian blur (radius=3, sigma=1.5) |
| 2 | `*_box_blur.ppm` | Box blur (radius=2) |
| 3 | `*_sharpened.ppm` | Sharpened image (strength=1.5) |
| 4 | `*_grayscale.pgm` | Grayscale conversion |
| 5 | `*_sobel_edges.pgm` | Sobel edge detection |
| 6 | `*_canny_edges.pgm` | Canny edge detection |
| 7 | `*_brightness_contrast.ppm` | Brightness/contrast adjusted |

## License

MIT
