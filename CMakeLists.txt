cmake_minimum_required(VERSION 3.13)
enable_language(CXX)

# require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(multi_targets CXX)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# main executable
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
target_compile_options(main PUBLIC -O3 -Wall)
