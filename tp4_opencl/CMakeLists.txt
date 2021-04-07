cmake_minimum_required (VERSION 3.16)
project(pp_tp4)

find_package(OpenCL REQUIRED)

set(CMAKE_CXX_STANDARD 20)

### Comment first line if building the 2 project in the same time
### Uncomment second line if building only openCL project
configure_file(inversion.cpp ${CMAKE_BINARY_DIR}/tp4_opencl COPYONLY)
#configure_file(inversion.cpp ${CMAKE_BINARY_DIR} COPYONLY)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")

add_executable(pp_opencl main.cpp Matrix.cpp)
target_link_libraries(pp_opencl OpenCL::OpenCL)