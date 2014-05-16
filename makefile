ifeq ($(PE_ENV),GNU)
	flags=-fopenmp
endif
ifeq ($(PE_ENV),CRAY)
	flags=
endif
ifeq ($(PE_ENV),INTEL)
	flags=-openmp
endif

all : affinity.omp affinity.mpi

affinity.omp: affinity_openmp.c
	cc $(flags) affinity_openmp.c -o affinity.omp

affinity.mpi: affinity_mpi.c
	cc $(flags) affinity_mpi.c -o affinity.mpi

clean:
	rm -rf affinity


