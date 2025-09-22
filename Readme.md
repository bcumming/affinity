# Affinity

Simple applications for determining Linux thread and gpu affinity.

There are examples for OpenMP and MPI+OpenMP.
* `affinity.omp` : for testing thread affinity without MPI
* `affinity.mpi` : for testing thread affinity of each rank in an MPI job
* `affinity.rocm` : for testing AMD GPU affinity of each rank an MPI job
* `affinity.cuda` : for testing NVIDIA GPU affinity of each rank in an MPI job

## Building

CMake is used to configure the build.

* `-DAFFINITY_MPI=[on,off]`: enable MPI support
    * default: `on`
* `-DAFFINITY_GPU_BACKEND=[none,cuda,rocm]`: enable a GPU backend
    * default: `none`
    * GPU backends require MPI support

Examples:
```
# build with only OpenMP (no MPI)
CC=gcc CXX=g++ cmake $source_path -DAFFINITY_MPI=off

# build with MPI support
CC=gcc CXX=g++ cmake $source_path

# build with cuda support
CC=gcc CXX=g++ cmake $source_path -DAFFINITY_GPU_BACKEND=cuda
```
