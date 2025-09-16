cmake_minimum_required(VERSION 3.14)
project(edge_infer_gateway LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(EIG_BUILD_SERVER "Build the TensorRT gateway server" ON)
option(EIG_BUILD_CLIENT_CPP "Build the C++ client" ON)

if(NOT EIG_BUILD_SERVER)
  message(STATUS "Server build disabled (CI mode).")
endif()

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
# TensorRT headers+libs are in NVIDIA containers and typical Ubuntu installs:
# include dir: /usr/include/x86_64-linux-gnu  lib: -lnvinfer -lnvonnxparser (optional)
include_directories(/usr/include/x86_64-linux-gnu)

# yaml-cpp via system package (apt install libyaml-cpp-dev)
find_package(yaml-cpp REQUIRED)

add_executable(edge-infer-gateway
  src/main.cpp
  src/gateway.cpp
  src/model_manager.cpp
  src/trt_runner.cpp
)
target_include_directories(edge-infer-gateway PRIVATE include)
target_link_libraries(edge-infer-gateway
  yaml-cpp
  Threads::Threads
  CUDA::cudart
  nvinfer
)

# nice warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(edge-infer-gateway PRIVATE -Wall -Wextra -Wno-deprecated-declarations)
endif()
