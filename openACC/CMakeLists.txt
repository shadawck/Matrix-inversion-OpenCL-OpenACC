cmake_minimum_required (VERSION 3.7)
project (pp_tp4)
set(CMAKE_CXX_STANDARD 20)

set(PG_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/21.2/compilers/bin)

set(CMAKE_C_COMPILER ${PG_PATH}/pgcc)
set(CMAKE_CXX_COMPILER "${PG_PATH}/pgc++")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fast -Minfo=all,accel,intensity,ccff  -Mcuda=cuda11.2 -ta=tesla:managed")

add_executable(pp_openacc main.cpp Matrix.cpp ${CMAKE_CXX_COMPILER})