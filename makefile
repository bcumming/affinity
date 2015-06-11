# the mpi compiler has to be set using an environment variable CC
flags=-fopenmp -std=c++11

all : test.omp test.mpi

test.omp: test_omp.cpp
	$(CC) $(flags) test_omp.cpp -o test.omp

test.mpi: test_mpi.cpp
	$(CC) $(flags) test_mpi.cpp -o test.mpi

clean:
	rm -rf test.omp
	rm -rf test.mpi


