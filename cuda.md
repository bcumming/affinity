# CUDA affinity

The CUDA affinity test is for seeing which GPUs are visible to each rank in an MPI job.
This is important when running on nodes with more than one GPU, because the user has to take care that GPUs are assigned to ranks in a sensible way for the application.

The `makefile` must be editted to add the `CUDAROOT` path (and maybe to set the `CC` variable to point to the compiler wrapper on your system). For example: `CUDAROOT=/usr/local/cuda-10.0`.

Then just:
```bash
make test.cuda
```

## Examples

### 2 Socket SkyLake with 4 V100 GPUS

```bash
# step 1: print out the unique identifiers of the GPUs on this node
bcumming@ault05.cscs.ch:affinity > nvidia-smi -q | grep UUID
    GPU UUID                        : GPU-4a6a534c-a207-2786-9db1-b893570f8251
    GPU UUID                        : GPU-672582ca-897e-01b6-69dd-61ed88ca0a87
    GPU UUID                        : GPU-3e335301-124b-15b2-5ce8-f7479e88924a
    GPU UUID                        : GPU-ef2b66cb-828e-abac-ff07-d4a0fe78a919

# step 2: run the test with default settings
#         note that all 4 ranks see all 4 GPUs
bcumming@ault05.cscs.ch:affinity > OMP_NUM_THREADS=1 mpirun -n 4 ./test.cuda
GPU affinity test for 4 MPI ranks
rank      0 @ ault05.cscs.ch
 gpu   0 : GPU-4a6a534c-a207-2786-9db1-b893570f8251
 gpu   1 : GPU-672582ca-897e-01b6-69dd-61ed88ca0a87
 gpu   2 : GPU-3e335301-124b-15b2-5ce8-f7479e88924a
 gpu   3 : GPU-ef2b66cb-828e-abac-ff07-d4a0fe78a919
rank      1 @ ault05.cscs.ch
 gpu   0 : GPU-4a6a534c-a207-2786-9db1-b893570f8251
 gpu   1 : GPU-672582ca-897e-01b6-69dd-61ed88ca0a87
 gpu   2 : GPU-3e335301-124b-15b2-5ce8-f7479e88924a
 gpu   3 : GPU-ef2b66cb-828e-abac-ff07-d4a0fe78a919
rank      2 @ ault05.cscs.ch
 gpu   0 : GPU-4a6a534c-a207-2786-9db1-b893570f8251
 gpu   1 : GPU-672582ca-897e-01b6-69dd-61ed88ca0a87
 gpu   2 : GPU-3e335301-124b-15b2-5ce8-f7479e88924a
 gpu   3 : GPU-ef2b66cb-828e-abac-ff07-d4a0fe78a919
rank      3 @ ault05.cscs.ch
 gpu   0 : GPU-4a6a534c-a207-2786-9db1-b893570f8251
 gpu   1 : GPU-672582ca-897e-01b6-69dd-61ed88ca0a87
 gpu   2 : GPU-3e335301-124b-15b2-5ce8-f7479e88924a
 gpu   3 : GPU-ef2b66cb-828e-abac-ff07-d4a0fe78a919

# step 3: hack together a script that assigns GPUs to ranks
#         note that this is for MVAPICH2
bcumming@ault05.cscs.ch:affinity > cat test.sh
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$MV2_COMM_WORLD_LOCAL_RANK
./test.cuda

# step 4: ... a unique GPU for each rank!
bcumming@ault05.cscs.ch:affinity > OMP_NUM_THREADS=1 mpirun -n 4 ./test.sh
GPU affinity test for 4 MPI ranks
rank      0 @ ault05.cscs.ch : GPU-4a6a534c-a207-2786-9db1-b893570f8251
rank      1 @ ault05.cscs.ch : GPU-672582ca-897e-01b6-69dd-61ed88ca0a87
rank      2 @ ault05.cscs.ch : GPU-3e335301-124b-15b2-5ce8-f7479e88924a
rank      3 @ ault05.cscs.ch : GPU-ef2b66cb-828e-abac-ff07-d4a0fe78a919
```

### 2 socket Power8 system with 2 GPUs

```bash
# step 1: print out the unique identifiers of the GPUs on this node
bcumming@openpower01:affinity > nvidia-smi -q | grep UUID
    GPU UUID                        : GPU-1beb496d-68a0-4095-7f57-1d1f58be2fcb
    GPU UUID                        : GPU-a0150e1c-a6ee-3f1b-75cf-6f60620ffc95

# step 2: run the test with default settings
#         note that all 2 ranks see all 2 GPUs
bcumming@openpower01:affinity > mpirun -n 2 test.cuda
GPU affinity test for 2 MPI ranks
rank      0 @ openpower01
 gpu   0 : GPU-1beb496d-68a0-4095-7f57-1d1f58be2fcb
 gpu   1 : GPU-a0150e1c-a6ee-3f1b-75cf-6f60620ffc95
rank      1 @ openpower01
 gpu   0 : GPU-1beb496d-68a0-4095-7f57-1d1f58be2fcb
 gpu   1 : GPU-a0150e1c-a6ee-3f1b-75cf-6f60620ffc95

# step 3: hack together a script that assigns GPUs to ranks
#         note that this is for OPENMPI
bcumming@openpower01:affinity > cat test.sh
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_NODE_RANK
./test.cuda

# step 4: ... a unique GPU for each rank!
bcumming@openpower01:affinity > mpirun -n 2 ./test.sh
GPU affinity test for 2 MPI ranks
rank      0 @ openpower01 : GPU-1beb496d-68a0-4095-7f57-1d1f58be2fcb
rank      1 @ openpower01 : GPU-a0150e1c-a6ee-3f1b-75cf-6f60620ffc95

```
