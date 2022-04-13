#include <iostream>
#include <vector>

#include <unistd.h>

#include <rocm_smi/rocm_smi.h>
#include <hip/hip_runtime_api.h>

#include "rocm.hpp"

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

