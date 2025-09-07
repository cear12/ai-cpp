#include "../face-detector.h"

#include <iostream>

FaceDetector::FaceDetector(const std::string& face_cascade_path, 
                          const std::string& eyes_cascade_path)
    : scale_factor(1.1), min_neighbors(3), min_size(30, 30) {
    
    if (!face_cascade.load(face_cascade_path)) {
        std::cerr << "Error loading face cascade: " << face_cascade_path << std::endl;
    }
    
    if (!eyes_cascade.load(eyes_cascade_path)) {
        std::cerr << "Error loading eyes cascade: " << eyes_cascade_path << std::endl;
    }
}

std::vector<cv::Rect> FaceDetector::detectFaces(const cv::Mat& image) {
    std::vector<cv::Rect> faces;
    cv::Mat gray;
    
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::equalizeHist(gray, gray);
    
    face_cascade.detectMultiScale(gray, faces, scale_factor, min_neighbors, 
                                 0 | cv::CASCADE_SCALE_IMAGE, min_size);
    
    return faces;
}

std::vector<cv::Rect> FaceDetector::detectEyes(const cv::Mat& face_roi) {
    std::vector<cv::Rect> eyes;
    cv::Mat gray_roi;
    
    if (face_roi.channels() == 3) {
        cv::cvtColor(face_roi, gray_roi, cv::COLOR_BGR2GRAY);
    } else {
        gray_roi = face_roi.clone();
    }
    
    eyes_cascade.detectMultiScale(gray_roi, eyes);
    
    return eyes;
}

void FaceDetector::drawDetections(cv::Mat& image, const std::vector<cv::Rect>& faces) {
    for (const auto& face : faces) {
        // Draw face rectangle
        cv::rectangle(image, face, cv::Scalar(255, 0, 0), 2);
        
        // Extract face ROI for eye detection
        cv::Mat face_roi = image(face);
        std::vector<cv::Rect> eyes = detectEyes(face_roi);
        
        // Draw eyes
        for (const auto& eye : eyes) {
            cv::Point center(face.x + eye.x + eye.width / 2, 
                           face.y + eye.y + eye.height / 2);
            int radius = cv::saturate_cast<int>((eye.width + eye.height) * 0.25);
            cv::circle(image, center, radius, cv::Scalar(0, 255, 0), 2);
        }
        
        // Add face count text
        cv::putText(image, "Face", 
                   cv::Point(face.x, face.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   cv::Scalar(255, 0, 0), 2);
    }
}

void FaceDetector::processVideo(const std::string& video_path) {
    cv::VideoCapture cap;
    
    if (video_path.empty()) {
        cap.open(0); // Default camera
    } else {
        cap.open(video_path);
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error opening video source" << std::endl;
        return;
    }
    
    cv::Mat frame;
    while (cap.read(frame)) {
        auto faces = detectFaces(frame);
        drawDetections(frame, faces);
        
        cv::imshow("Face Detection", frame);
        
        if (cv::waitKey(30) >= 0) break;
    }
    
    cv::destroyAllWindows();
}

bool FaceDetector::isInitialized() const {
    return !face_cascade.empty();
}
