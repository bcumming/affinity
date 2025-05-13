cmake_minimum_required(VERSION 3.21)
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
    add_executable(affinity.cuda gpu.cpp cuda.cpp)
    target_include_directories(affinity.cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(affinity.cuda PRIVATE AFFINITY_CUDA)
    target_link_libraries(affinity.cuda PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX CUDA::cudart)
endif()

if(DEFINED ENV{ROCM_PATH} AND MPI_FOUND)
    set(ROCM_PATH $ENV{ROCM_PATH})
    set(ROCM TRUE)
else()
    message("Unable to congifure ROCM target")
endif()

if (ROCM)
    find_package(rocm_smi REQUIRED PATHS "${ROCM_PATH}/lib/cmake/rocm_smi")
    find_package(hip REQUIRED PATHS "${ROCM_PATH}/lib/cmake/hip")
    add_executable(affinity.rocm gpu.cpp rocm.cpp)
    target_compile_definitions(affinity.rocm PRIVATE __HIP_PLATFORM_AMD__)
    target_compile_definitions(affinity.rocm PRIVATE AFFINITY_ROCM)
    target_include_directories(affinity.rocm PRIVATE "${ROCM_SMI_INCLUDE_DIR}" "${HIP_INCLUDE_DIR}")
    target_link_libraries(affinity.rocm PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
    target_link_libraries(affinity.rocm PRIVATE ${ROCM_SMI_LIBRARIES})
    target_link_libraries(affinity.rocm PRIVATE hip::host)
endif()

# Install rules
install(TARGETS affinity.omp DESTINATION bin)
if(MPI_FOUND)
    install(TARGETS affinity.mpi DESTINATION bin)
endif()
if(CUDA)
    install(TARGETS affinity.cuda DESTINATION bin)
endif()
if(ROCM)
    install(TARGETS affinity.rocm DESTINATION bin)
endif()
