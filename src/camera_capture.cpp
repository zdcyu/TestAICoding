#include "camera_capture.h"

#include <iostream>

CameraCapture::CameraCapture() : running_(false) {}

CameraCapture::~CameraCapture() {
    close();
}

bool CameraCapture::open(int deviceIndex) {
    cap_.open(deviceIndex, cv::CAP_ANY);
    if (!cap_.isOpened()) {
        std::cerr << "Error: Cannot open camera device " << deviceIndex << std::endl;
        return false;
    }
    std::cout << "Camera opened: device " << deviceIndex
              << " (" << getWidth() << "x" << getHeight()
              << " @ " << getFPS() << " FPS)" << std::endl;
    return true;
}

bool CameraCapture::open(const std::string& source) {
    cap_.open(source);
    if (!cap_.isOpened()) {
        std::cerr << "Error: Cannot open video source: " << source << std::endl;
        return false;
    }
    std::cout << "Video source opened: " << source
              << " (" << getWidth() << "x" << getHeight()
              << " @ " << getFPS() << " FPS)" << std::endl;
    return true;
}

void CameraCapture::close() {
    running_ = false;
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool CameraCapture::isOpened() const {
    return cap_.isOpened();
}

bool CameraCapture::readFrame(cv::Mat& frame) {
    if (!cap_.isOpened()) return false;
    return cap_.read(frame);
}

void CameraCapture::setResolution(int width, int height) {
    if (cap_.isOpened()) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
}

int CameraCapture::getWidth() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int CameraCapture::getHeight() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double CameraCapture::getFPS() const {
    return cap_.get(cv::CAP_PROP_FPS);
}

void CameraCapture::run(const std::string& windowName, FrameCallback callback) {
    if (!cap_.isOpened()) {
        std::cerr << "Error: Camera not opened. Call open() first." << std::endl;
        return;
    }

    running_ = true;
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    while (running_) {
        if (!cap_.read(frame) || frame.empty()) {
            std::cerr << "Warning: Failed to read frame." << std::endl;
            break;
        }

        // Call the user callback; stop if it returns false
        if (callback && !callback(frame)) {
            break;
        }

        // Display the frame
        cv::imshow(windowName, frame);

        // Check for quit keys (q, Q, ESC)
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    running_ = false;
    cv::destroyWindow(windowName);
}

void CameraCapture::stop() {
    running_ = false;
}
