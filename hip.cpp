#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <mpi.h>
#include <unistd.h>

#include <rocm_smi/rocm_smi.h>
#include <hip/hip_runtime_api.h>

// Store uid in a byte array for easy type punning and comparison.
// uuids follow the most common storage format, which is big-endian.
struct alignas(8) uuid {
    std::array<unsigned char, 8> bytes;

    uuid() {
        std::fill(bytes.begin(), bytes.end(), 0);
    }

    uuid(const uuid&) = default;

    uuid(uint64_t i) {
        const unsigned char* b = reinterpret_cast<const unsigned char*>(&i);
        std::copy(b, b+sizeof(uint64_t), bytes.begin());
    }
};

// Test GPU uids for equality
bool operator==(const uuid& lhs, const uuid& rhs) {
    for (auto i=0u; i<lhs.bytes.size(); ++i) {
        if (lhs.bytes[i]!=rhs.bytes[i]) return false;
    }
    return true;
}

// Strict lexographical ordering of GPU uids
bool operator<(const uuid& lhs, const uuid& rhs) {
    for (auto i=0u; i<lhs.bytes.size(); ++i) {
        if (lhs.bytes[i]<rhs.bytes[i]) return true;
        if (lhs.bytes[i]>lhs.bytes[i]) return false;
    }
    return false;
}

std::ostream& operator<<(std::ostream& o, const uuid& id) {
    o << std::hex << std::setfill('0');

    for (int i=0; i<8; ++i) {
        o << std::setw(2) << (int)id.bytes[7-i];
    }
    o << std::dec;
    return o;
}


void check_mpi_call(int status) {
    if(status!=MPI_SUCCESS) {
        std::cerr << "Error in MPI" << std::endl;
        exit(1);
    }
}

std::string get_hostname() {
    const int maxlen = 128;
    char name[maxlen];
    if( gethostname(name, maxlen) ) {
        std::cerr << "Error finding host name" << std::endl;
        exit(1);
    }

    return std::string(name);
}

std::vector<uuid> get_gpu_uuids() {
    // get number of devices
    int ngpus = 0;
    // TODO: proper error checking on api calls
    if (hipGetDeviceCount(&ngpus)) return {};

    uint64_t init_flags=0;
    if (rsmi_init(init_flags)) {
        std::cout << "ERROR: unable to initialize rsmi\n";
        exit(1);
    }

    // store the uuids
    std::vector<uuid> uuids(ngpus);

    // using CUDA 10 or later, determining uuid of GPUs is easy!
    for (int i=0; i<ngpus; ++i) {
        uint64_t uuid;
        if (rsmi_dev_unique_id_get(i, &uuid)) {
            std::cout << "ERROR: rsmi_dev_unique_id_get\n";
            exit(1);
        }
        uuids[i] = uuid;
    }

    rsmi_shut_down();

    return uuids;
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

