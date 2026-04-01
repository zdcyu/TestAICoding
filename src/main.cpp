#include "image_processor.h"

#include <iostream>
#include <string>

void printUsage() {
    std::cout << "CUDA Computer Vision - Image Processing Demo" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: CudaComputerVision [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --input <file>     Input PPM/PGM image file" << std::endl;
    std::cout << "  --output <prefix>  Output file prefix (default: output)" << std::endl;
    std::cout << "  --width <N>        Test image width  (default: 1920)" << std::endl;
    std::cout << "  --height <N>       Test image height (default: 1080)" << std::endl;
    std::cout << "  --device <id>      CUDA device ID    (default: 0)" << std::endl;
    std::cout << "  --help             Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "If no input file is specified, a gradient test image is generated." << std::endl;
}

int main(int argc, char* argv[]) {
    std::string inputFile;
    std::string outputPrefix = "output";
    int width = 1920;
    int height = 1080;
    int deviceId = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
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
        }
    }

    std::cout << "=== CUDA Computer Vision ===" << std::endl;
    std::cout << std::endl;

    // Initialize the image processor
    ImageProcessor processor;
    if (!processor.initialize(deviceId)) {
        std::cerr << "Failed to initialize CUDA device " << deviceId << std::endl;
        return 1;
    }
    processor.printDeviceInfo();
    std::cout << std::endl;

    // Load or generate input image
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

    // 1. Gaussian Blur
    std::cout << "[1/6] Gaussian Blur (radius=3, sigma=1.5)..." << std::endl;
    Image blurred = processor.gaussianBlur(inputImage, 3, 1.5f);
    ImageProcessor::saveImagePPM(blurred, outputPrefix + "_gaussian_blur.ppm");
    std::cout << std::endl;

    // 2. Box Blur
    std::cout << "[2/6] Box Blur (radius=2)..." << std::endl;
    Image boxBlurred = processor.boxBlur(inputImage, 2);
    ImageProcessor::saveImagePPM(boxBlurred, outputPrefix + "_box_blur.ppm");
    std::cout << std::endl;

    // 3. Sharpen
    std::cout << "[3/6] Sharpen (strength=1.5)..." << std::endl;
    Image sharpened = processor.sharpen(inputImage, 1.5f);
    ImageProcessor::saveImagePPM(sharpened, outputPrefix + "_sharpened.ppm");
    std::cout << std::endl;

    // 4. Grayscale Conversion
    std::cout << "[4/6] RGB to Grayscale..." << std::endl;
    Image grayscale = processor.rgbToGrayscale(inputImage);
    ImageProcessor::saveImagePPM(grayscale, outputPrefix + "_grayscale.pgm");
    std::cout << std::endl;

    // 5. Sobel Edge Detection
    std::cout << "[5/6] Sobel Edge Detection (threshold=0.1)..." << std::endl;
    Image sobelEdges = processor.sobelEdgeDetection(inputImage, 0.1f);
    ImageProcessor::saveImagePPM(sobelEdges, outputPrefix + "_sobel_edges.pgm");
    std::cout << std::endl;

    // 6. Canny Edge Detection
    std::cout << "[6/6] Canny Edge Detection (low=0.05, high=0.15)..." << std::endl;
    Image cannyEdges = processor.cannyEdgeDetection(inputImage, 0.05f, 0.15f);
    ImageProcessor::saveImagePPM(cannyEdges, outputPrefix + "_canny_edges.pgm");
    std::cout << std::endl;

    // 7. Brightness/Contrast Adjustment
    std::cout << "[Bonus] Brightness/Contrast (brightness=+30, contrast=1.2)..." << std::endl;
    Image adjusted = processor.brightnessContrast(inputImage, 30.0f, 1.2f);
    ImageProcessor::saveImagePPM(adjusted, outputPrefix + "_brightness_contrast.ppm");
    std::cout << std::endl;

    std::cout << "=== All processing complete! ===" << std::endl;
    std::cout << "Output files saved with prefix: " << outputPrefix << std::endl;

    return 0;
}
