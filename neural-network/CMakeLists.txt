cmake_minimum_required(VERSION 3.10)
project(neural-network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(neural_network
    src/main.cpp
    src/neural-network.cpp
)

# Enable compiler optimizations for release build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(neural_network PRIVATE -O3)
endif()

# Enable debug symbols for debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(neural_network PRIVATE -g -Wall -Wextra)
endif()
