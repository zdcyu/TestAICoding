#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

// Detection result for a single object
struct Detection {
    int classId;            // Class index
    std::string className;  // Human-readable class name
    float confidence;       // Detection confidence [0, 1]
    cv::Rect boundingBox;   // Bounding box in image coordinates
    cv::Scalar color;       // Display color for this class
};

// YOLO-based object detector using OpenCV DNN with optional CUDA backend
class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector();

    // Load a YOLO model (ONNX format)
    // modelPath:  path to .onnx model file (e.g. yolov5s.onnx, yolov8n.onnx)
    // classesPath: path to text file with class names (one per line, e.g. coco.names)
    // useCuda:    use CUDA backend for GPU acceleration (requires OpenCV built with CUDA)
    bool loadModel(const std::string& modelPath,
                   const std::string& classesPath,
                   bool useCuda = true);

    // Detect objects in an image
    // confThreshold: minimum confidence to keep a detection
    // nmsThreshold:  Non-Maximum Suppression IoU threshold
    std::vector<Detection> detect(const cv::Mat& frame,
                                  float confThreshold = 0.45f,
                                  float nmsThreshold = 0.50f);

    // Draw detection results on an image
    static void drawDetections(cv::Mat& frame,
                               const std::vector<Detection>& detections,
                               bool showConfidence = true);

    // Check if model is loaded
    bool isLoaded() const;

    // Get class names
    const std::vector<std::string>& getClassNames() const;

    // Set input size for the network (default 640x640 for YOLOv5/v8)
    void setInputSize(int width, int height);

private:
    cv::dnn::Net net_;
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar> classColors_;
    bool loaded_;
    int inputWidth_;
    int inputHeight_;

    // Load class names from a text file
    bool loadClassNames(const std::string& path);

    // Generate distinct colors for each class
    void generateColors();

    // Pre-process input image for YOLO
    cv::Mat preprocess(const cv::Mat& frame);

    // Post-process YOLO output to extract detections (YOLOv5/v8 format)
    std::vector<Detection> postprocess(const cv::Mat& frame,
                                       const std::vector<cv::Mat>& outputs,
                                       float confThreshold,
                                       float nmsThreshold);
};
