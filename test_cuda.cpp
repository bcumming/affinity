#include <iomanip>
#include <iostream>
#include <sstream>

#include <mpi.h>
#include <unistd.h>

#include "gpu.hpp"

void check_mpi_call(int status) {
    if(status!=MPI_SUCCESS) {
        std::cerr << "error in MPI" << std::endl;
        exit(1);
    }
}

std::string get_hostname() {
    const int maxlen = 128;
    char name[maxlen];
    if( gethostname(name, maxlen) ) {
        std::cerr << "error finding host name" << std::endl;
        exit(1);
    }

    return std::string(name);
}

int main(int argc, char **argv) {

    check_mpi_call( MPI_Init(&argc, &argv));

    int mpi_rank;
    int mpi_size;
    check_mpi_call( MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    check_mpi_call( MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    const int mpi_root = 0;

    auto hostname = get_hostname();

    auto gpus = get_gpu_uuids();
    auto num_gpus = gpus.size();

    if(mpi_rank==mpi_root) {
        std::cout << "GPU affinity test for " << mpi_size << " MPI ranks\n";
    }

    std::stringstream s;
    s << "rank " << std::setw(6) << mpi_rank << " @ " << hostname;
    if(num_gpus==1) {
        s << " : GPU-" << gpus[0] << "\n";
    }
    else {
        s << "\n";
        for(auto i=0; i<num_gpus; ++i) {
            s << " gpu " << std::setfill(' ') << std::setw(3) << i
              << " : GPU-" << gpus[i]
              << "\n";
        }
    }
    auto message = s.str();
    // add 1 for the terminating \0
    int message_length = message.size() + 1;

    std::vector<int> message_lengths(mpi_size);
    check_mpi_call(
        MPI_Gather( &message_length,     1, MPI_INT,
                    &message_lengths[0], 1, MPI_INT,
                    mpi_root, MPI_COMM_WORLD)
    );

    if(mpi_rank==mpi_root) {
        std::cout << message;
        for(auto i=1; i<mpi_size; ++i) {
            std::vector<char> remote_msg(message_lengths[i]);
            MPI_Status status;
            check_mpi_call(
                MPI_Recv( &remote_msg[0], message_lengths[i], MPI_CHAR,
                          i, i, MPI_COMM_WORLD, &status)
            );
            std::cout << &remote_msg[0];
        }
    }
    else {
        check_mpi_call(
            MPI_Send( message.c_str(), message_length, MPI_CHAR,
                      mpi_root, mpi_rank, MPI_COMM_WORLD)
        );
    }

    MPI_Finalize();
}

