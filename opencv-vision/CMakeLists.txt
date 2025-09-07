cmake_minimum_required(VERSION 3.10)
project(opencv_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(opencv_vision
    src/main.cpp
    src/face-detector.cpp
    src/object-tracker.cpp
)

# Link OpenCV libraries
target_link_libraries(opencv_vision ${OpenCV_LIBS})

# Copy data files to build directory
file(COPY data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(opencv_vision PRIVATE -O3)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(opencv_vision PRIVATE -g -Wall -Wextra)
endif()
