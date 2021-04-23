cmake_minimum_required(VERSION 3.14)
project(affinity)

# Generate a json file with compilation commands for each file
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ settings
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(affinity.omp test_omp.cpp)
target_link_libraries(affinity.omp PRIVATE OpenMP::OpenMP_CXX)

find_package(MPI)
if(MPI_FOUND)
    add_executable(affinity.mpi test_mpi.cpp)
    target_link_libraries(affinity.mpi PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
endif()
