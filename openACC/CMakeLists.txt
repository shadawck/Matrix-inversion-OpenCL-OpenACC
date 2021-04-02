cmake_minimum_required (VERSION 3.7)
project (pp_tp4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS " -fopenacc -foffload=nvptx-none -flto -Wall -Wextra")

find_package(OpenMP REQUIRED)

add_executable(pp_openacc main.cpp Matrix.cpp)


INCLUDE(CPack)
