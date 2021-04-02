cmake_minimum_required (VERSION 3.16)
project(pp_tp4)

find_package(OpenCL REQUIRED)

set(CMAKE_CXX_STANDARD 20)
configure_file(VectorAdd.cl ${CMAKE_BINARY_DIR}/openCL COPYONLY)

add_executable(pp_opencl VectorAdd.cpp)
target_link_libraries(pp_opencl OpenCL::OpenCL)