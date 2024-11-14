cmake_minimum_required(VERSION 3.10)
project(clip_trt)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include 디렉토리 설정
include_directories(
    ${PROJECT_SOURCE_DIR}/  # 현재 프로젝트 폴더
    /usr/local/include/opencv4  # OpenCV 헤더 파일 경로
    /usr/local/cuda/include     # CUDA 헤더 파일 경로
    /usr/include/x86_64-linux-gnu  # 시스템 경로
    /home/jihoon-epitone/Downloads/install_lenovo_part2-001/librealsense/include  # RealSense 경로
)

# 라이브러리 경로 설정
link_directories(
    /usr/local/lib
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
)

# 실행 파일 설정
add_executable(clip_trt clip_trt.cpp)

# 필요한 라이브러리 설정
target_link_libraries(clip_trt
    opencv_core
    opencv_imgcodecs
    opencv_highgui
    opencv_imgproc
    opencv_video  # KLT Tracker를 사용하기 위한 추가
    opencv_videoio  # 비디오 입출력을 위한 추가
    cudart
    nvinfer
    nvinfer_plugin
    nvonnxparser
    realsense2
    opencv_dnn
    opencv_tracking # 추가: 트래킹 관련 라이브러리
)

# 디버그 모드 빌드 설정 (gdb 사용 가능)
set(CMAKE_BUILD_TYPE Debug)

# 컴파일러 옵션
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always -g")

