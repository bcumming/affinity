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
else()
    message("Unable to configure MPI target")
endif()

find_package(CUDAToolkit 10)
if (CUDAToolkit_FOUND AND MPI_FOUND)
    set(CUDA TRUE)
else()
    message("Unable to configure CUDA target")
endif()

if(CUDA)
    enable_language(CUDA)
    add_executable(affinity.gpu gpu.cpp cuda.cpp)
    target_include_directories(affinity.gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(affinity.gpu PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX CUDA::cudart)
endif()

if(DEFINED ENV{ROCM_PATH} AND MPI_FOUND)
    set(ROCM_PATH $ENV{ROCM_PATH})
    set(ROCM TRUE)
else()
    message("Unable to congifure ROCM target")
endif()

if (ROCM)
    set(SMI_PATH "${ROCM_PATH}/rocm_smi")
    set(HIP_PATH "${ROCM_PATH}/hip")
    add_executable(affinity.hip hip.cpp)
    target_compile_definitions(affinity.hip PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(affinity.hip PRIVATE "${SMI_PATH}/include/" "${HIP_PATH}/include")
    target_link_libraries(affinity.hip PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
    target_link_libraries(affinity.hip PRIVATE -L${SMI_PATH}/lib -lrocm_smi64)
    target_link_libraries(affinity.hip PRIVATE -L${HIP_PATH}/lib -lamdhip64)
endif()
