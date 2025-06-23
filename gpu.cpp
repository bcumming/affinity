#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <mpi.h>
#include <omp.h>
#include <unistd.h>

#include "affinity.h"
#ifdef AFFINITY_CUDA
#include "cuda.hpp"
#endif
#ifdef AFFINITY_ROCM
#include "rocm.hpp"
#endif
#include "util.h"

void check_mpi_call(int status) {
    if (status != MPI_SUCCESS) {
        std::cerr << "Error in MPI" << std::endl;
        exit(1);
    }
}

std::string get_hostname() {
    const int maxlen = 128;
    char name[maxlen];
    if (gethostname(name, maxlen)) {
        std::cerr << "Error finding host name" << std::endl;
        exit(1);
    }

    return std::string(name);
}

int main(int argc, char** argv) {

    check_mpi_call(MPI_Init(&argc, &argv));

    int mpi_rank;
    int mpi_size;
    check_mpi_call(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    check_mpi_call(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    const int mpi_root = 0;

    // get hostname
    auto hostname = get_hostname();
    // get gpu information
    auto gpus = get_gpu_uuids();
    auto num_gpus = gpus.size();
    // get core affinity information
    auto num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> per_thread_cores(num_threads);
#pragma omp parallel
    {
        per_thread_cores[omp_get_thread_num()] = get_affinity();
    }
    std::vector<int> all_cores;
    for (auto& cores : per_thread_cores) {
        all_cores.insert(all_cores.end(), cores.begin(), cores.end());
    }
    std::sort(all_cores.begin(), all_cores.end());
    auto back = std::unique(all_cores.begin(), all_cores.end());
    all_cores.erase(back, all_cores.end());

    if (mpi_rank == mpi_root) {
        std::cout << "GPU affinity test for " << mpi_size << " MPI ranks\n";
    }

    std::stringstream s;
    // print rank header
    s << "rank " << std::setw(6) << mpi_rank << " @ " << hostname << "\n";

    // print condensed summary of cpu affinity
    s << " cores   : " << print_range(all_cores, 0) << "\n";

    // print gpu identification one per line
    for (auto i = 0; i < num_gpus; ++i) {
        s << " gpu " << std::setfill(' ') << std::setw(3) << i << " : GPU-"
          << gpus[i] << "\n";
    }

    auto message = s.str();
    // add 1 for the terminating \0
    int message_length = message.size() + 1;

    std::vector<int> message_lengths(mpi_size);
    check_mpi_call(MPI_Gather(&message_length, 1, MPI_INT, &message_lengths[0],
                              1, MPI_INT, mpi_root, MPI_COMM_WORLD));

    if (mpi_rank == mpi_root) {
        std::cout << message;
        for (auto i = 1; i < mpi_size; ++i) {
            std::vector<char> remote_msg(message_lengths[i]);
            MPI_Status status;
            check_mpi_call(MPI_Recv(&remote_msg[0], message_lengths[i],
                                    MPI_CHAR, i, i, MPI_COMM_WORLD, &status));
            std::cout << &remote_msg[0];
        }
    } else {
        check_mpi_call(MPI_Send(message.c_str(), message_length, MPI_CHAR,
                                mpi_root, mpi_rank, MPI_COMM_WORLD));
    }

    MPI_Finalize();
}
