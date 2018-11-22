#include <algorithm>
#include <array>
#include <vector>
#include <iomanip>
#include <ostream>

#include <cuda_runtime.h>

#include "gpu.hpp"

// Test GPU uids for equality
bool operator==(const uuid& lhs, const uuid& rhs) {
    for (auto i=0u; i<sizeof(cudaUUID_t); ++i) {
        if (lhs.bytes[i]!=rhs.bytes[i]) return false;
    }
    return true;
}

// Strict lexographical ordering of GPU uids
bool operator<(const uuid& lhs, const uuid& rhs) {
    for (auto i=0u; i<sizeof(cudaUUID_t); ++i) {
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

std::vector<uuid> get_gpu_uuids() {
    // get number of devices
    int ngpus = 0;
    auto status = cudaGetDeviceCount(&ngpus);

    // store the uuids
    std::vector<uuid> uuids(ngpus);
    for (int i=0; i<ngpus; ++i) {
        cudaDeviceProp props;
        auto status = cudaGetDeviceProperties(&props, i);
        uuids[i] = props.uuid;
    }

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

