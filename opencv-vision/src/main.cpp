#include "../face-detector.h"
#include "../object-tracker.h"

#include <iostream>
#include <string>

void showMenu() {
    std::cout << "\n=== Computer Vision Demo ===" << std::endl;
    std::cout << "1. Face Detection (Camera)" << std::endl;
    std::cout << "2. Face Detection (Video File)" << std::endl;
    std::cout << "3. Object Tracking (Interactive)" << std::endl;
    std::cout << "4. Edge Detection Demo" << std::endl;
    std::cout << "0. Exit" << std::endl;
    std::cout << "Choose option: ";
}

void edgeDetectionDemo() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return;
    }
    
    cv::Mat frame, gray, edges, blurred;
    
    std::cout << "Edge Detection Demo - Press ESC to exit" << std::endl;
    
    while (cap.read(frame)) {
        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Apply Gaussian blur
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);
        
        // Apply Canny edge detection
        cv::Canny(blurred, edges, 50, 150);
        
        // Convert back to BGR for display
        cv::Mat edges_colored;
        cv::cvtColor(edges, edges_colored, cv::COLOR_GRAY2BGR);
        
        // Create side-by-side comparison
        cv::Mat comparison;
        cv::hconcat(frame, edges_colored, comparison);
        
        cv::imshow("Original vs Edges", comparison);
        
        if (cv::waitKey(30) == 27) break; // ESC
    }
    
    cv::destroyAllWindows();
}

int main() {
    int choice;
    
    while (true) {
        showMenu();
        std::cin >> choice;
        
        switch (choice) {
            case 1: {
                FaceDetector detector;
                if (detector.isInitialized()) {
                    detector.processVideo();
                } else {
                    std::cout << "Failed to initialize face detector" << std::endl;
                }
                break;
            }
            
            case 2: {
                std::string video_path;
                std::cout << "Enter video file path: ";
                std::cin >> video_path;
                
                FaceDetector detector;
                if (detector.isInitialized()) {
                    detector.processVideo(video_path);
                } else {
                    std::cout << "Failed to initialize face detector" << std::endl;
                }
                break;
            }
            
            case 3: {
                ObjectTracker tracker;
                tracker.runInteractiveTracking();
                break;
            }
            
            case 4: {
                edgeDetectionDemo();
                break;
            }
            
            case 0:
                std::cout << "Exiting..." << std::endl;
                return 0;
                
            default:
                std::cout << "Invalid choice. Try again." << std::endl;
        }
    }
    
    return 0;
}
