#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <vector>
#include <memory>

class ObjectTracker {
private:
    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<cv::Rect2d> bounding_boxes;
    std::vector<cv::Scalar> colors;
    cv::Mat previous_frame;
    std::vector<std::vector<cv::Point2f>> tracks;
    
    cv::Scalar generateRandomColor();

public:
    ObjectTracker();
    
    // Multi-object tracking
    void addTracker(const cv::Mat& frame, const cv::Rect2d& bbox);
    void updateTrackers(const cv::Mat& frame);
    void drawTrackingResults(cv::Mat& frame);
    void clearTrackers();
    
    // Optical flow tracking
    void initOpticalFlow(const cv::Mat& frame, const std::vector<cv::Point2f>& points);
    void updateOpticalFlow(const cv::Mat& frame);
    void drawOpticalFlow(cv::Mat& frame);
    
    // Interactive tracking demo
    void runInteractiveTracking(const std::string& video_path = "");
    
    size_t getTrackerCount() const { return trackers.size(); }
};
