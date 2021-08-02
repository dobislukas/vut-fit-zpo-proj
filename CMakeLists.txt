#Lukas Dobis 
#xdobis01
#ZPO 2020/2021
#Image segmentation based on optical flow

project(ZPO_PROJECT_XDOBIS01)
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( segmentVideo src/main.cpp )
target_link_libraries( segmentVideo ${OpenCV_LIBS})
