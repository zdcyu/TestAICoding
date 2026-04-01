# CUDA Computer Vision

A C++ project using NVIDIA CUDA for GPU-accelerated computer vision and image processing. Includes a Visual Studio solution (`.sln` / `.vcxproj`) for Windows development and CMake support for cross-platform builds.

## Features

- **Real-Time Object Detection** - Camera-based object recognition using YOLO (YOLOv5/v8) with CUDA-accelerated inference via OpenCV DNN
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
│   ├── image_processor.h            # High-level ImageProcessor class
│   ├── camera_capture.h             # Camera/video capture wrapper (OpenCV)
│   └── object_detector.h            # YOLO object detector (OpenCV DNN)
├── src/
│   ├── main.cpp                     # Entry point (process / detect modes)
│   ├── image_processor.cpp          # ImageProcessor implementation
│   ├── camera_capture.cpp           # CameraCapture implementation
│   └── object_detector.cpp          # ObjectDetector implementation
├── kernels/
│   ├── image_filters.cu             # Gaussian blur, box blur, sharpen kernels
│   ├── color_conversion.cu          # RGB-Grayscale, RGB-HSV, brightness kernels
│   ├── edge_detection.cu            # Sobel & Canny edge detection kernels
│   └── detection_preprocessing.cu   # Resize, letterbox, BGR-to-RGB kernels
├── data/
│   └── coco.names                   # COCO dataset class names (80 classes)
└── build/                           # CMake build output (gitignored)
```

## Requirements

- **NVIDIA GPU** with Compute Capability 7.5+ (Turing / Ampere / Ada Lovelace)
- **CUDA Toolkit 12.0+** ([download](https://developer.nvidia.com/cuda-downloads))
- **OpenCV 4.x** with DNN module ([download](https://opencv.org/releases/))
  - For CUDA-accelerated inference, build OpenCV with CUDA and cuDNN support
- **C++17** compatible compiler

### Windows (Visual Studio)

- Visual Studio 2022 (v143 toolset)
- CUDA Toolkit 12.0 integrated with Visual Studio
- OpenCV 4.x (set `OPENCV_DIR` environment variable to your OpenCV install path)

### Linux / macOS (CMake)

- CMake 3.18+
- GCC 9+ or Clang 10+
- CUDA Toolkit 12.0+
- OpenCV 4.x (`sudo apt install libopencv-dev` on Ubuntu)

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

The program has two modes: **process** (image processing pipeline) and **detect** (real-time camera object detection).

### Image Processing Mode

```bash
# Run with auto-generated test image (1920x1080)
./bin/CudaComputerVision process

# Run with custom test image size
./bin/CudaComputerVision process --width 3840 --height 2160

# Run with an input PPM image
./bin/CudaComputerVision process --input myimage.ppm --output results
```

### Object Detection Mode

First, download a YOLO model in ONNX format:

```bash
# YOLOv8 nano (fastest, recommended for real-time)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx -P data/

# Or YOLOv5 small
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx -P data/
```

Then run detection:

```bash
# Real-time camera detection with CUDA acceleration
./bin/CudaComputerVision detect --model data/yolov8n.onnx

# Detect objects in a video file
./bin/CudaComputerVision detect --model data/yolov8n.onnx --video test.mp4

# Use a different camera
./bin/CudaComputerVision detect --model data/yolov8n.onnx --camera 1

# Use CPU-only inference (no CUDA required)
./bin/CudaComputerVision detect --model data/yolov8n.onnx --no-cuda

# Adjust detection sensitivity
./bin/CudaComputerVision detect --model data/yolov8n.onnx --conf 0.3 --nms 0.4
```

Press **q** or **ESC** in the detection window to stop.

### Image Processing Output

The `process` mode runs a full pipeline and saves results as PPM/PGM files:

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
