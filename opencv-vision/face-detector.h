#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

class FaceDetector {
private:
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;
    double scale_factor;
    int min_neighbors;
    cv::Size min_size;

public:
    FaceDetector(const std::string& face_cascade_path = "data/haarcascade_frontalface_alt.xml",
                 const std::string& eyes_cascade_path = "data/haarcascade_eye_tree_eyeglasses.xml");
    
    std::vector<cv::Rect> detectFaces(const cv::Mat& image);
    std::vector<cv::Rect> detectEyes(const cv::Mat& face_roi);
    void drawDetections(cv::Mat& image, const std::vector<cv::Rect>& faces);
    void processVideo(const std::string& video_path = "");
    bool isInitialized() const;
    
    // Configuration methods
    void setScaleFactor(double scale) { scale_factor = scale; }
    void setMinNeighbors(int neighbors) { min_neighbors = neighbors; }
    void setMinSize(const cv::Size& size) { min_size = size; }
};
