cmake_minimum_required (VERSION 3.6)
project (pp_tp4)

include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -Wno-unused-variable -Wshadow -Wuninitialized -Wunused-label")

add_executable(pp_openacc main.cpp Matrix.cpp)
target_link_libraries(pp_openacc)

INCLUDE(CPack)