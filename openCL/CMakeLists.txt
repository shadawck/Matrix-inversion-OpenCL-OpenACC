cmake_minimum_required (VERSION 3.16)
project(pp_tp4)

find_package(OpenCL REQUIRED)

set(CMAKE_CXX_STANDARD 20)
configure_file(inversion.cpp ${CMAKE_BINARY_DIR}/openCL COPYONLY)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")

add_executable(pp_opencl main.cpp Matrix.cpp)
target_link_libraries(pp_opencl OpenCL::OpenCL)