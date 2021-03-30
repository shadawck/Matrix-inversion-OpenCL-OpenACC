cmake_minimum_required (VERSION 3.16)
project(pp_tp4 CXX)

find_package(OpenCL REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(pp_tp4 VectorAdd.cpp)
target_link_libraries(pp_tp4 OpenCL::OpenCL)