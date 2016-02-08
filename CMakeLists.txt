cmake_minimum_required(VERSION 2.8)
project(diff_filter_emu)
add_executable(diff_filter_emu diff_filter_emu.cpp)

set(OpenCV_DIR "C:/dev/opencv-3.1.0/build/install")

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  target_link_libraries(diff_filter_emu ${OpenCV_LIBS})
endif(OpenCV_FOUND)