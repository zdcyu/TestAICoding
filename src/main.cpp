#include "image_processor.h"
#include "camera_capture.h"
#include "object_detector.h"

#include <iostream>
#include <string>

void printUsage() {
    std::cout << "CUDA Computer Vision - Image Processing & Object Detection" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: CudaComputerVision [mode] [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Modes:" << std::endl;
    std::cout << "  process             Run image processing pipeline (default)" << std::endl;
    std::cout << "  detect              Run real-time camera object detection" << std::endl;
    std::cout << std::endl;
    std::cout << "Image Processing Options:" << std::endl;
    std::cout << "  --input <file>      Input PPM/PGM image file" << std::endl;
    std::cout << "  --output <prefix>   Output file prefix (default: output)" << std::endl;
    std::cout << "  --width <N>         Test image width  (default: 1920)" << std::endl;
    std::cout << "  --height <N>        Test image height (default: 1080)" << std::endl;
    std::cout << std::endl;
    std::cout << "Object Detection Options:" << std::endl;
    std::cout << "  --model <file>      YOLO ONNX model file (e.g. yolov5s.onnx)" << std::endl;
    std::cout << "  --classes <file>    Class names file (default: data/coco.names)" << std::endl;
    std::cout << "  --camera <id>       Camera device index (default: 0)" << std::endl;
    std::cout << "  --video <file>      Video file to process instead of camera" << std::endl;
    std::cout << "  --conf <threshold>  Confidence threshold (default: 0.45)" << std::endl;
    std::cout << "  --nms <threshold>   NMS IoU threshold (default: 0.50)" << std::endl;
    std::cout << "  --no-cuda           Use CPU instead of CUDA for inference" << std::endl;
    std::cout << "  --input-size <N>    Model input size (default: 640)" << std::endl;
    std::cout << std::endl;
    std::cout << "Common Options:" << std::endl;
    std::cout << "  --device <id>       CUDA device ID (default: 0)" << std::endl;
    std::cout << "  --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  CudaComputerVision process --input photo.ppm" << std::endl;
    std::cout << "  CudaComputerVision detect --model yolov8n.onnx" << std::endl;
    std::cout << "  CudaComputerVision detect --model yolov5s.onnx --video test.mp4" << std::endl;
    std::cout << "  CudaComputerVision detect --model yolov8n.onnx --camera 1 --no-cuda" << std::endl;
}

// Run the image processing pipeline
int runImageProcessing(const std::string& inputFile, const std::string& outputPrefix,
                       int width, int height, int deviceId) {
    ImageProcessor processor;
    if (!processor.initialize(deviceId)) {
        std::cerr << "Failed to initialize CUDA device " << deviceId << std::endl;
        return 1;
    }
    processor.printDeviceInfo();
    std::cout << std::endl;

    Image inputImage;
    if (!inputFile.empty()) {
        inputImage = ImageProcessor::loadImagePPM(inputFile);
        if (inputImage.width == 0) {
            std::cerr << "Failed to load image: " << inputFile << std::endl;
            return 1;
        }
    } else {
        std::cout << "Generating test image (" << width << "x" << height << ")..." << std::endl;
        inputImage = ImageProcessor::generateTestImage(width, height);
        ImageProcessor::saveImagePPM(inputImage, outputPrefix + "_original.ppm");
    }

    std::cout << std::endl;
    std::cout << "--- Running Image Processing Pipeline ---" << std::endl;
    std::cout << std::endl;

    std::cout << "[1/6] Gaussian Blur (radius=3, sigma=1.5)..." << std::endl;
    Image blurred = processor.gaussianBlur(inputImage, 3, 1.5f);
    ImageProcessor::saveImagePPM(blurred, outputPrefix + "_gaussian_blur.ppm");
    std::cout << std::endl;

    std::cout << "[2/6] Box Blur (radius=2)..." << std::endl;
    Image boxBlurred = processor.boxBlur(inputImage, 2);
    ImageProcessor::saveImagePPM(boxBlurred, outputPrefix + "_box_blur.ppm");
    std::cout << std::endl;

    std::cout << "[3/6] Sharpen (strength=1.5)..." << std::endl;
    Image sharpened = processor.sharpen(inputImage, 1.5f);
    ImageProcessor::saveImagePPM(sharpened, outputPrefix + "_sharpened.ppm");
    std::cout << std::endl;

    std::cout << "[4/6] RGB to Grayscale..." << std::endl;
    Image grayscale = processor.rgbToGrayscale(inputImage);
    ImageProcessor::saveImagePPM(grayscale, outputPrefix + "_grayscale.pgm");
    std::cout << std::endl;

    std::cout << "[5/6] Sobel Edge Detection (threshold=0.1)..." << std::endl;
    Image sobelEdges = processor.sobelEdgeDetection(inputImage, 0.1f);
    ImageProcessor::saveImagePPM(sobelEdges, outputPrefix + "_sobel_edges.pgm");
    std::cout << std::endl;

    std::cout << "[6/6] Canny Edge Detection (low=0.05, high=0.15)..." << std::endl;
    Image cannyEdges = processor.cannyEdgeDetection(inputImage, 0.05f, 0.15f);
    ImageProcessor::saveImagePPM(cannyEdges, outputPrefix + "_canny_edges.pgm");
    std::cout << std::endl;

    std::cout << "[Bonus] Brightness/Contrast (brightness=+30, contrast=1.2)..." << std::endl;
    Image adjusted = processor.brightnessContrast(inputImage, 30.0f, 1.2f);
    ImageProcessor::saveImagePPM(adjusted, outputPrefix + "_brightness_contrast.ppm");
    std::cout << std::endl;

    std::cout << "=== All processing complete! ===" << std::endl;
    std::cout << "Output files saved with prefix: " << outputPrefix << std::endl;
    return 0;
}

// Run camera-based object detection
int runObjectDetection(const std::string& modelPath, const std::string& classesPath,
                       int cameraId, const std::string& videoFile,
                       float confThreshold, float nmsThreshold,
                       bool useCuda, int inputSize) {
    if (modelPath.empty()) {
        std::cerr << "Error: --model is required for object detection mode." << std::endl;
        std::cerr << "Usage: CudaComputerVision detect --model <path_to_yolo.onnx>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Download a YOLO model:" << std::endl;
        std::cerr << "  YOLOv8n: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx" << std::endl;
        std::cerr << "  YOLOv5s: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx" << std::endl;
        return 1;
    }

    // Load the detector
    ObjectDetector detector;
    detector.setInputSize(inputSize, inputSize);

    std::cout << "Loading model: " << modelPath << std::endl;
    if (!detector.loadModel(modelPath, classesPath, useCuda)) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }
    std::cout << std::endl;

    // Open the camera or video source
    CameraCapture camera;
    if (!videoFile.empty()) {
        if (!camera.open(videoFile)) return 1;
    } else {
        if (!camera.open(cameraId)) return 1;
    }

    std::cout << std::endl;
    std::cout << "=== Real-Time Object Detection ===" << std::endl;
    std::cout << "Press 'q' or ESC to quit." << std::endl;
    std::cout << std::endl;

    // Frame counter for FPS display
    int frameCount = 0;
    double totalTime = 0.0;

    camera.run("CUDA Object Detection", [&](cv::Mat& frame) -> bool {
        auto startTime = std::chrono::high_resolution_clock::now();

        // Detect objects
        std::vector<Detection> detections = detector.detect(frame, confThreshold, nmsThreshold);

        auto endTime = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(endTime - startTime).count();

        totalTime += elapsed;
        frameCount++;

        // Draw detections
        ObjectDetector::drawDetections(frame, detections);

        // Draw FPS and detection count
        double fps = 1000.0 / elapsed;
        double avgFps = frameCount * 1000.0 / totalTime;

        char fpsText[128];
        snprintf(fpsText, sizeof(fpsText), "FPS: %.1f (avg: %.1f) | Objects: %zu",
                 fps, avgFps, detections.size());

        cv::putText(frame, fpsText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Print detections to console every 30 frames
        if (frameCount % 30 == 0 && !detections.empty()) {
            std::cout << "[Frame " << frameCount << "] Detected " << detections.size() << " object(s):";
            for (const auto& det : detections) {
                std::cout << " " << det.className << "(" << static_cast<int>(det.confidence * 100) << "%)";
            }
            std::cout << std::endl;
        }

        return true;
    });

    std::cout << std::endl;
    std::cout << "=== Detection Complete ===" << std::endl;
    std::cout << "Total frames: " << frameCount << std::endl;
    if (frameCount > 0) {
        std::cout << "Average FPS: " << (frameCount * 1000.0 / totalTime) << std::endl;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    std::string mode = "process";
    std::string inputFile;
    std::string outputPrefix = "output";
    std::string modelPath;
    std::string classesPath = "data/coco.names";
    std::string videoFile;
    int width = 1920;
    int height = 1080;
    int deviceId = 0;
    int cameraId = 0;
    float confThreshold = 0.45f;
    float nmsThreshold = 0.50f;
    bool useCuda = true;
    int inputSize = 640;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else if (arg == "process" || arg == "detect") {
            mode = arg;
        } else if (arg == "--input" && i + 1 < argc) {
            inputFile = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputPrefix = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            deviceId = std::stoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--classes" && i + 1 < argc) {
            classesPath = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            cameraId = std::stoi(argv[++i]);
        } else if (arg == "--video" && i + 1 < argc) {
            videoFile = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            confThreshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            nmsThreshold = std::stof(argv[++i]);
        } else if (arg == "--no-cuda") {
            useCuda = false;
        } else if (arg == "--input-size" && i + 1 < argc) {
            inputSize = std::stoi(argv[++i]);
        }
    }

    std::cout << "=== CUDA Computer Vision ===" << std::endl;
    std::cout << std::endl;

    if (mode == "detect") {
        return runObjectDetection(modelPath, classesPath, cameraId, videoFile,
                                  confThreshold, nmsThreshold, useCuda, inputSize);
    } else {
        return runImageProcessing(inputFile, outputPrefix, width, height, deviceId);
    }
}
