#include "../object-tracker.h"

#include <iostream>
#include <random>

ObjectTracker::ObjectTracker() {}

cv::Scalar ObjectTracker::generateRandomColor() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    return cv::Scalar(dis(gen), dis(gen), dis(gen));
}

void ObjectTracker::addTracker(const cv::Mat& frame, const cv::Rect2d& bbox) {
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    
    if (tracker->init(frame, bbox)) {
        trackers.push_back(tracker);
        bounding_boxes.push_back(bbox);
        colors.push_back(generateRandomColor());
        std::cout << "Tracker " << trackers.size() << " initialized" << std::endl;
    }
}

void ObjectTracker::updateTrackers(const cv::Mat& frame) {
    for (size_t i = 0; i < trackers.size(); ++i) {
        cv::Rect2d bbox;
        if (trackers[i]->update(frame, bbox)) {
            bounding_boxes[i] = bbox;
        } else {
            std::cout << "Tracking failed for tracker " << i << std::endl;
        }
    }
}

void ObjectTracker::drawTrackingResults(cv::Mat& frame) {
    for (size_t i = 0; i < bounding_boxes.size(); ++i) {
        cv::rectangle(frame, bounding_boxes[i], colors[i], 2);
        
        std::string label = "Object " + std::to_string(i + 1);
        cv::putText(frame, label,
                   cv::Point(bounding_boxes[i].x, bounding_boxes[i].y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2);
    }
}

void ObjectTracker::clearTrackers() {
    trackers.clear();
    bounding_boxes.clear();
    colors.clear();
    tracks.clear();
}

void ObjectTracker::initOpticalFlow(const cv::Mat& frame, 
                                   const std::vector<cv::Point2f>& points) {
    cv::cvtColor(frame, previous_frame, cv::COLOR_BGR2GRAY);
    tracks.clear();
    tracks.push_back(points);
}

void ObjectTracker::updateOpticalFlow(const cv::Mat& frame) {
    if (previous_frame.empty() || tracks.empty()) return;
    
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::Point2f> next_points;
    std::vector<uchar> status;
    std::vector<float> error;
    
    cv::calcOpticalFlowPyrLK(previous_frame, gray_frame, tracks.back(),
                            next_points, status, error);
    
    // Keep only good points
    std::vector<cv::Point2f> good_points;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] == 1) {
            good_points.push_back(next_points[i]);
        }
    }
    
    if (!good_points.empty()) {
        tracks.push_back(good_points);
    }
    
    // Keep only recent tracks (last 30 frames)
    if (tracks.size() > 30) {
        tracks.erase(tracks.begin());
    }
    
    gray_frame.copyTo(previous_frame);
}

void ObjectTracker::drawOpticalFlow(cv::Mat& frame) {
    if (tracks.size() < 2) return;
    
    std::vector<cv::Scalar> colors_flow = {
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)
    };
    
    // Draw tracks
    for (size_t i = 1; i < tracks.size(); ++i) {
        for (size_t j = 0; j < tracks[i].size() && j < tracks[i-1].size(); ++j) {
            cv::line(frame, tracks[i-1][j], tracks[i][j], 
                    colors_flow[j % colors_flow.size()], 2);
        }
    }
    
    // Draw current points
    if (!tracks.empty()) {
        for (size_t i = 0; i < tracks.back().size(); ++i) {
            cv::circle(frame, tracks.back()[i], 3, 
                      colors_flow[i % colors_flow.size()], -1);
        }
    }
}

void ObjectTracker::runInteractiveTracking(const std::string& video_path) {
    cv::VideoCapture cap;
    
    if (video_path.empty()) {
        cap.open(0);
    } else {
        cap.open(video_path);
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error opening video source" << std::endl;
        return;
    }
    
    cv::Mat frame;
    bool selecting = false;
    cv::Rect2d selection_box;
    
    std::cout << "Instructions:" << std::endl;
    std::cout << "- Press 's' to select object for tracking" << std::endl;
    std::cout << "- Press 'c' to clear all trackers" << std::endl;
    std::cout << "- Press 'ESC' to exit" << std::endl;
    
    while (cap.read(frame)) {
        if (!trackers.empty()) {
            updateTrackers(frame);
            drawTrackingResults(frame);
        }
        
        cv::imshow("Object Tracking", frame);
        
        char key = cv::waitKey(30) & 0xFF;
        if (key == 27) break; // ESC
        
        if (key == 's') {
            selection_box = cv::selectROI("Object Tracking", frame, false);
            if (selection_box.width > 0 && selection_box.height > 0) {
                addTracker(frame, selection_box);
            }
        }
        
        if (key == 'c') {
            clearTrackers();
            std::cout << "All trackers cleared" << std::endl;
        }
    }
    
    cv::destroyAllWindows();
}
