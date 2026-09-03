cmake_minimum_required(VERSION 3.10)
project(opencv_vision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# NOTE: this repo's CMake setup has not been run locally -- OpenCV isn't
# installable without root in the sandbox this was cleaned up in (see the
# repository README). CI is the first real compiler/linker this sees.
find_package(OpenCV REQUIRED)

add_executable(opencv_vision
    src/main.cpp
    src/face-detector.cpp
    src/object-tracker.cpp
)
target_include_directories(opencv_vision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_vision PRIVATE ${OpenCV_LIBS})
target_compile_options(opencv_vision PRIVATE -Wall -Wextra -Wpedantic)

# FaceDetector's default constructor arguments point at data/<cascade>.xml
# (relative to the working directory the binary is run from). The Haar
# cascade XML files themselves are NOT vendored in this repo (they're
# freely redistributable, but sizeable, and every OpenCV install already
# ships them -- see the README for where to get them). If a data/
# directory has been populated locally, copy it next to the built binary;
# if not, skip this step instead of failing the CMake configure (the
# original version of this file unconditionally did `file(COPY data/ ...)`,
# which errors out at configure time when data/ doesn't exist -- and it
# never existed anywhere in this repo's history).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)
else()
    message(STATUS "opencv-vision: no data/ directory found -- FaceDetector's default cascade paths "
                    "won't resolve unless you pass your own paths or populate data/ (see README).")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(opencv_vision PRIVATE -O3)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(opencv_vision PRIVATE -g)
endif()
