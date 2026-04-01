#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <functional>
#include <string>
#include <atomic>

// Camera capture wrapper with real-time frame callback
class CameraCapture {
public:
    // Callback type: receives a frame and returns true to continue, false to stop
    using FrameCallback = std::function<bool(cv::Mat&)>;

    CameraCapture();
    ~CameraCapture();

    // Open camera by device index (0 = default camera)
    bool open(int deviceIndex = 0);

    // Open camera by URL (e.g. RTSP stream, video file)
    bool open(const std::string& source);

    // Close the camera
    void close();

    // Check if camera is opened
    bool isOpened() const;

    // Get a single frame
    bool readFrame(cv::Mat& frame);

    // Set camera resolution
    void setResolution(int width, int height);

    // Get camera properties
    int getWidth() const;
    int getHeight() const;
    double getFPS() const;

    // Run capture loop with a callback for each frame
    // Press 'q' or ESC in the OpenCV window to stop
    void run(const std::string& windowName, FrameCallback callback);

    // Stop the capture loop
    void stop();

private:
    cv::VideoCapture cap_;
    std::atomic<bool> running_;
};
