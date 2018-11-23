#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <vector>

#include <cuda_runtime.h>

#if CUDART_VERSION < 10000
    #define USE_NVML
    #include <nvml.h>
#endif

#include "gpu.hpp"

#include <iostream>

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

    bool first = true;
    int ranges[6] = {0, 4, 6, 8, 10, 16};
    for (int i=0; i<5; ++i) { // because std::size isn't available till C++17
        if (!first) o << "-";
        for (auto j=ranges[i]; j<ranges[i+1]; ++j) {
            o << std::setw(2) << (int)id.bytes[j];
        }
        first = false;
    }
    o << std::dec;
    return o;
}

uuid string_to_uuid(char* str) {
    uuid result;
    unsigned n = std::strlen(str);
    if (n!=40) {
        std::cout << "expected length 40 uuid string, got: " << n << "\n";
        return result;
    }

    // Converts a single hex character, i.e. 0123456789abcdef, to int
    // Assumes that input is a valid hex character.
    auto hex_c2i = [](char c) -> unsigned char {
        return std::isalpha(c)? c-'a'+10: c-'0';
    };

    // This removes the "GPU" from front of string, and the '-' hyphens:
    //      GPU-f1fd7811-e4d3-4d54-abb7-efc579fb1e28
    // becomes
    //      f1fd7811e4d34d54abb7efc579fb1e28
    auto pos = std::remove_if(
            str, str+n, [](char c){return !std::isxdigit(c);});
    n = pos-str;

    // null terminate the shortened string
    str[n] = 0;

    // convert pairs of characters into single bytes.
    for (int i=0; i<16; ++i) {
        const char* s = str+2*i;
        result.bytes[i] = (hex_c2i(s[0])<<4) + hex_c2i(s[1]);
    }

    return result;
}

std::vector<uuid> get_gpu_uuids() {
    // get number of devices
    int ngpus = 0;
    auto status = cudaGetDeviceCount(&ngpus);

    // store the uuids
    std::vector<uuid> uuids(ngpus);
#ifdef USE_NVML
    auto nvml_status = nvmlInit(); // TODO: can we init?
    //std::cout << nvmlErrorString(nvml_status) << "\n";
    char buffer[41];
    for (int i=0; i<ngpus; ++i) {
        // get handle of gpu with index i
        nvmlDevice_t handle;
        nvml_status = nvmlDeviceGetHandleByIndex(i, &handle); // TODO: can we get the GPU
        // get uuid as a string with format GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        nvml_status =  nvmlDeviceGetUUID(handle, buffer, 128);
        uuids[i] = string_to_uuid(buffer);
    }
    nvml_status = nvmlShutdown();
#else
    for (int i=0; i<ngpus; ++i) {
        cudaDeviceProp props;
        auto status = cudaGetDeviceProperties(&props, i);
        uuids[i] = props.uuid;
    }
#endif

    return uuids;
}

using uuid_range = std::pair<std::vector<uuid>::const_iterator,
                               std::vector<uuid>::const_iterator>;

std::ostream& operator<<(std::ostream& o, uuid_range rng) {
    o << "[";
    for (auto i=rng.first; i!=rng.second; ++i) {
        o << " " << int(i->bytes[0]);
    }
    return o << "]";
}


// Compare two sets of uuids
//   1: both sets are identical
//  -1: some common elements
//   0: no common elements
int compare_gpu_groups(uuid_range l, uuid_range r) {
    auto range_size = [] (uuid_range rng) { return std::distance(rng.first, rng.second);};
    if (range_size(l)<range_size(r)) {
        std::swap(l, r);
    }

    unsigned count = 0;
    for (auto it=l.first; it!=l.second; ++it) {
        if (std::find(r.first, r.second, *it)!=r.second) ++count;
    }

    // test for complete match
    if (count==range_size(l) && count==range_size(r)) return 1;
    // test for partial match
    if (count) return -1;
    return 0;
}

gpu_rank assign_gpu(const std::vector<uuid>& uids,
               const std::vector<int>&    uid_part,
               int rank)
{
    // Determine the number of ranks in MPI communicator
    auto nranks = uid_part.size()-1;

    // Helper that generates the range of gpu id for rank i
    auto make_group = [&] (int i) {
        return uuid_range{uids.begin()+uid_part[i], uids.begin()+uid_part[i+1]};
    };

    // The list of ranks that share the same GPUs as this rank (including this rank).
    std::vector<int> neighbors;

    // Indicate if an invalid GPU partition was encountered.
    bool error = false;

    // The gpu uid range for this rank
    auto local_gpus = make_group(rank);

    // Find all ranks with the same set of GPUs as this rank.
    for (std::size_t i=0; i<nranks; ++i) {
        auto other_gpus = make_group(i);
        auto match = compare_gpu_groups(local_gpus, other_gpus);
        if (match==1) { // found a match
            neighbors.push_back(i);
        }
        else if (match==-1) { // partial match, which is not permitted
            error=true;
            break;
        }
        // case where match==0 can be ignored.
    }

    if (error) {
        return gpu_status::error;
    }

    // Determine the position of this rank in the sorted list of ranks.
    auto pos_in_group =
        std::distance(
            neighbors.begin(),
            std::find(neighbors.begin(), neighbors.end(), rank));

    // The number of GPUs available to the ranks.
    auto ngpu_in_group = std::distance(local_gpus.first, local_gpus.second);

    // Assign GPUs to the first ngpu ranks. If there are more ranks than GPUs,
    // some ranks will not be assigned a GPU (return -1).
    return pos_in_group<ngpu_in_group? gpu_rank(pos_in_group): gpu_rank(gpu_status::none);
}

