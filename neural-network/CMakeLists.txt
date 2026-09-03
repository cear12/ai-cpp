cmake_minimum_required(VERSION 3.10)
project(neural-network CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(neural_network
    src/main.cpp
    src/neural-network.cpp
)
target_include_directories(neural_network PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_compile_options(neural_network PRIVATE /W4)
else()
    target_compile_options(neural_network PRIVATE -Wall -Wextra -Wpedantic)

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(neural_network PRIVATE -O3)
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(neural_network PRIVATE -g)
    endif()
endif()
