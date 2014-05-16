ifeq ($(PE_ENV),GNU)
	flags=-fopenmp
endif
ifeq ($(PE_ENV),CRAY)
	flags=
endif
ifeq ($(PE_ENV),INTEL)
	flags=-openmp
endif

affinity: affinity_openmp.c
	cc $(flags) affinity_openmp.c -o affinity

clean:
	rm -rf affinity


