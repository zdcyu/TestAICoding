#include "object_detector.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>

ObjectDetector::ObjectDetector()
    : loaded_(false), inputWidth_(640), inputHeight_(640) {}

ObjectDetector::~ObjectDetector() = default;

bool ObjectDetector::loadModel(const std::string& modelPath,
                               const std::string& classesPath,
                               bool useCuda) {
    // Load class names
    if (!loadClassNames(classesPath)) {
        std::cerr << "Error: Failed to load class names from " << classesPath << std::endl;
        return false;
    }

    // Load ONNX model
    try {
        net_ = cv::dnn::readNetFromONNX(modelPath);
    } catch (const cv::Exception& e) {
        std::cerr << "Error: Failed to load ONNX model: " << e.what() << std::endl;
        return false;
    }

    // Set backend and target
    if (useCuda) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "Using CUDA backend for inference." << std::endl;
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "Using CPU backend for inference." << std::endl;
    }

    generateColors();
    loaded_ = true;

    std::cout << "Model loaded: " << modelPath << std::endl;
    std::cout << "Classes: " << classNames_.size() << std::endl;
    std::cout << "Input size: " << inputWidth_ << "x" << inputHeight_ << std::endl;

    return true;
}

std::vector<Detection> ObjectDetector::detect(const cv::Mat& frame,
                                              float confThreshold,
                                              float nmsThreshold) {
    if (!loaded_) {
        std::cerr << "Error: Model not loaded. Call loadModel() first." << std::endl;
        return {};
    }

    // Preprocess
    cv::Mat blob = preprocess(frame);

    // Forward pass
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    // Postprocess
    return postprocess(frame, outputs, confThreshold, nmsThreshold);
}

void ObjectDetector::drawDetections(cv::Mat& frame,
                                    const std::vector<Detection>& detections,
                                    bool showConfidence) {
    for (const auto& det : detections) {
        // Draw bounding box
        cv::rectangle(frame, det.boundingBox, det.color, 2);

        // Build label
        std::string label = det.className;
        if (showConfidence) {
            char buf[32];
            snprintf(buf, sizeof(buf), " %.1f%%", det.confidence * 100.0f);
            label += buf;
        }

        // Draw label background
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                              0.6, 1, &baseLine);
        int top = std::max(det.boundingBox.y, labelSize.height);
        cv::rectangle(frame,
                      cv::Point(det.boundingBox.x, top - labelSize.height - 4),
                      cv::Point(det.boundingBox.x + labelSize.width, top + 2),
                      det.color, cv::FILLED);

        // Draw label text
        cv::putText(frame, label,
                    cv::Point(det.boundingBox.x, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

bool ObjectDetector::isLoaded() const {
    return loaded_;
}

const std::vector<std::string>& ObjectDetector::getClassNames() const {
    return classNames_;
}

void ObjectDetector::setInputSize(int width, int height) {
    inputWidth_ = width;
    inputHeight_ = height;
}

bool ObjectDetector::loadClassNames(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    classNames_.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Remove trailing whitespace/carriage return
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
                line.pop_back();
            }
            classNames_.push_back(line);
        }
    }

    return !classNames_.empty();
}

void ObjectDetector::generateColors() {
    classColors_.clear();
    // Generate visually distinct colors using HSV
    for (size_t i = 0; i < classNames_.size(); ++i) {
        float hue = static_cast<float>(i) / classNames_.size() * 180.0f;
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(static_cast<int>(hue), 200, 255));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        auto* p = rgb.ptr<uint8_t>(0);
        classColors_.emplace_back(p[0], p[1], p[2]);
    }
}

cv::Mat ObjectDetector::preprocess(const cv::Mat& frame) {
    // Letterbox resize: maintain aspect ratio, pad to inputWidth_ x inputHeight_
    return cv::dnn::blobFromImage(frame, 1.0 / 255.0,
                                  cv::Size(inputWidth_, inputHeight_),
                                  cv::Scalar(0, 0, 0),
                                  true,   // swapRB (BGR -> RGB)
                                  false); // crop
}

std::vector<Detection> ObjectDetector::postprocess(const cv::Mat& frame,
                                                    const std::vector<cv::Mat>& outputs,
                                                    float confThreshold,
                                                    float nmsThreshold) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    int imgWidth = frame.cols;
    int imgHeight = frame.rows;

    // Scale factors for converting from model coordinates to image coordinates
    float xScale = static_cast<float>(imgWidth) / inputWidth_;
    float yScale = static_cast<float>(imgHeight) / inputHeight_;

    // The output format depends on the YOLO version:
    // YOLOv5: [batch, num_detections, 5 + num_classes]  (x_center, y_center, w, h, obj_conf, class_scores...)
    // YOLOv8: [batch, 4 + num_classes, num_detections]  (x_center, y_center, w, h, class_scores...)

    const cv::Mat& output = outputs[0];

    // Determine format based on output shape
    // output.size[1] vs output.size[2]
    int rows = output.size[1];
    int cols = output.size[2];

    bool isYoloV8 = false;

    // YOLOv8: cols == num_detections (e.g. 8400), rows == 4 + num_classes
    // YOLOv5: rows == num_detections, cols == 5 + num_classes
    if (rows < cols && rows == static_cast<int>(4 + classNames_.size())) {
        isYoloV8 = true;
    }

    if (isYoloV8) {
        // YOLOv8 format: [1, 4+num_classes, num_detections]
        // Transpose to [num_detections, 4+num_classes]
        cv::Mat transposed;
        cv::Mat reshaped(rows, cols, CV_32F, (void*)output.ptr<float>());
        cv::transpose(reshaped, transposed);

        int numDetections = transposed.rows;
        int numClasses = static_cast<int>(classNames_.size());

        for (int i = 0; i < numDetections; ++i) {
            const float* row = transposed.ptr<float>(i);

            // Find best class score
            float maxScore = 0.0f;
            int maxClassId = 0;
            for (int c = 0; c < numClasses; ++c) {
                float score = row[4 + c];
                if (score > maxScore) {
                    maxScore = score;
                    maxClassId = c;
                }
            }

            if (maxScore >= confThreshold) {
                float cx = row[0] * xScale;
                float cy = row[1] * yScale;
                float w  = row[2] * xScale;
                float h  = row[3] * yScale;

                int left   = static_cast<int>(cx - w / 2.0f);
                int top    = static_cast<int>(cy - h / 2.0f);
                int width  = static_cast<int>(w);
                int height = static_cast<int>(h);

                classIds.push_back(maxClassId);
                confidences.push_back(maxScore);
                boxes.emplace_back(left, top, width, height);
            }
        }
    } else {
        // YOLOv5 format: [1, num_detections, 5 + num_classes]
        int numDetections = rows;
        int numClasses = cols - 5;

        for (int i = 0; i < numDetections; ++i) {
            const float* row = output.ptr<float>(0, i);
            float objConf = row[4];

            if (objConf < confThreshold) continue;

            // Find best class
            float maxScore = 0.0f;
            int maxClassId = 0;
            for (int c = 0; c < numClasses; ++c) {
                float score = row[5 + c] * objConf;
                if (score > maxScore) {
                    maxScore = score;
                    maxClassId = c;
                }
            }

            if (maxScore >= confThreshold) {
                float cx = row[0] * xScale;
                float cy = row[1] * yScale;
                float w  = row[2] * xScale;
                float h  = row[3] * yScale;

                int left   = static_cast<int>(cx - w / 2.0f);
                int top    = static_cast<int>(cy - h / 2.0f);
                int width  = static_cast<int>(w);
                int height = static_cast<int>(h);

                classIds.push_back(maxClassId);
                confidences.push_back(maxScore);
                boxes.emplace_back(left, top, width, height);
            }
        }
    }

    // Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // Build final detections
    std::vector<Detection> detections;
    detections.reserve(indices.size());

    for (int idx : indices) {
        Detection det;
        det.classId = classIds[idx];
        det.confidence = confidences[idx];
        det.boundingBox = boxes[idx];

        if (det.classId >= 0 && det.classId < static_cast<int>(classNames_.size())) {
            det.className = classNames_[det.classId];
            det.color = classColors_[det.classId];
        } else {
            det.className = "unknown";
            det.color = cv::Scalar(128, 128, 128);
        }

        detections.push_back(det);
    }

    return detections;
}
