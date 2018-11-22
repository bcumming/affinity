# the mpi compiler has to be set using an environment variable CC
flags=-fopenmp -std=c++11
CC=mpicxx

# set this to the base path
CUDAROOT=
cudaflags=${flags} -L${CUDAROOT}/lib64 -I${CUDAROOT}/include -lcudart

all : test.omp test.mpi

test.omp: test_omp.cpp
	$(CC) $(flags) test_omp.cpp -o test.omp

test.mpi: test_mpi.cpp
	$(CC) $(flags) test_mpi.cpp -o test.mpi

test.cuda: test_cuda.cpp gpu.o
	${CC} ${cudaflags} gpu.o test_cuda.cpp -o test.cuda

gpu.o: gpu.cpp gpu.hpp
	${CC} ${cudaflags} -c gpu.cpp

clean:
	rm -rf test.omp
	rm -rf test.mpi
	rm -rf test.cuda
	rm -rf *.o


