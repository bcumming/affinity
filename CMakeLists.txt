cmake_minimum_required(VERSION 3.17)
project(affinity)

# Generate a json file with compilation commands for each file
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ settings
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(affinity.omp omp.cpp)
target_link_libraries(affinity.omp PRIVATE OpenMP::OpenMP_CXX)

find_package(MPI)
if(MPI_FOUND)
    add_executable(affinity.mpi mpi.cpp)
    target_link_libraries(affinity.mpi PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
endif()

find_package(CUDAToolkit 10)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_executable(affinity.gpu gpu.cpp cuda.cpp)
    target_include_directories(affinity.gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(affinity.gpu PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX CUDA::cudart)
endif()
