#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

#include <cuda_runtime.h>

#include "gpu.hpp"

#ifdef USE_NVML
    #include <nvml.h>
#endif

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

// Split CUDA_VISIBLE_DEVICES variable string into a list of integers.
// The environment variable can have spaces, and the order is important:
// i.e. "0,1" is not the same as "1,0".
//      CUDA_VISIBLE_DEVICES="1,0"
//      CUDA_VISIBLE_DEVICES="0, 1"
// The CUDA run time parses the list until it finds an error, then returns
// the partial list.
// i.e.
//      CUDA_VISIBLE_DEVICES="1, 0, hello" -> {1}
//      CUDA_VISIBLE_DEVICES="hello, 1" -> {}
// All non-numeric characters appear to be ignored:
//      CUDA_VISIBLE_DEVICES="0a,1" -> {0,1}
// This doesn't try too hard to check for all possible errors.
std::vector<int> parse_visible_devices(std::string str, unsigned ngpu) {
    // Tokenize into a sequence of strings separated by commas
    std::vector<std::string> strings;
    std::size_t first = 0;
    std::size_t last = str.find(',');
    while (last != std::string::npos) {
        strings.push_back(str.substr(first, last - first));
        first = last + 1;
        last = str.find(',', first);
    }
    strings.push_back(str.substr(first, last - first));

    // Convert each token to an integer.
    // Return partial list of ids on first error:
    //  - error converting token to string;
    //  - invalid GPU id found.
    std::vector<int> values;
    for (auto& s: strings) {
        try {
            int v = std::stoi(s);
            if (v<0 || v>=ngpu) break;
            values.push_back(v);
        }
        catch (std::exception e) {
            break;
        }
    }

    return values;
}

// Take a uuid string with the format:
//      GPU-f1fd7811-e4d3-4d54-abb7-efc579fb1e28
// And convert to a 16 byte sequence
//
// Assume that the intput string is correctly formatted.
uuid string_to_uuid(char* str) {
    uuid result;
    unsigned n = std::strlen(str);

    // Remove the "GPU" from front of string, and the '-' hyphens, e.g.:
    //      GPU-f1fd7811-e4d3-4d54-abb7-efc579fb1e28
    // becomes
    //      f1fd7811e4d34d54abb7efc579fb1e28
    auto pos = std::remove_if(
            str, str+n, [](char c){return !std::isxdigit(c);});

    // Converts a single hex character, i.e. 0123456789abcdef, to int
    // Assumes that input is a valid hex character.
    auto hex_c2i = [](char c) -> unsigned char {
        return std::isalpha(c)? c-'a'+10: c-'0';
    };

    // Convert pairs of characters into single bytes.
    for (int i=0; i<16; ++i) {
        const char* s = str+2*i;
        result.bytes[i] = (hex_c2i(s[0])<<4) + hex_c2i(s[1]);
    }

    return result;
}


#ifdef USE_NVML
// On error of any kind, return an empty list.
std::vector<uuid> get_gpu_uuids() {
    // Get number of devices.
    int ngpus = 0;
    if (cudaGetDeviceCount(&ngpus)!=cudaSuccess) return {};

    // Attempt to initialize nvml
    if (nvmlInit()!=NVML_SUCCESS) return {};

    // store the uuids
    std::vector<uuid> uuids;

    // find the number of available GPUs
    unsigned count = -1;
    if (nvmlDeviceGetCount(&count)!=NVML_SUCCESS) {
        nvmlShutdown();
        return {};
    }

    // Test if the environment variable CUDA_VISIBLE_DEVICES has been set.
    const char* visible_device_env = std::getenv("CUDA_VISIBLE_DEVICES");
    std::vector<int> device_ids;
    // If set, attempt to parse the device ids from it.
    if (visible_device_env) {
        // Parse the gpu ids from the environment variable
        device_ids = parse_visible_devices(visible_device_env, count);
        if ((unsigned)ngpus != device_ids.size()) {
            // Mismatch between device count detected by cuda runtime
            // and that set in environment variable.
            nvmlShutdown();
            return {};
        }
    }
    // Not set, so all devices must be available.
    else {
        device_ids.resize(count);
        std::iota(device_ids.begin(), device_ids.end(), 0);
    }

    // For each device id, query NVML for the device's uuid.
    for (int i: device_ids) {
        char buffer[41];
        // get handle of gpu with index i
        nvmlDevice_t handle;
        if (nvmlDeviceGetHandleByIndex(i, &handle)!=NVML_SUCCESS)
            goto on_error;

        // get uuid as a string with format GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        if (nvmlDeviceGetUUID(handle, buffer, sizeof(buffer))!=NVML_SUCCESS)
            goto on_error;

        uuids.push_back(string_to_uuid(buffer));
    }
    nvmlShutdown();

    return uuids;

on_error:
    nvmlShutdown();
    return {};
}

#else

std::vector<uuid> get_gpu_uuids() {
    // get number of devices
    int ngpus = 0;
    if (cudaGetDeviceCount(&ngpus)!=cudaSuccess) return {};

    // store the uuids
    std::vector<uuid> uuids(ngpus);
    // using CUDA 10 or later, determining uuid of GPUs is easy!
    for (int i=0; i<ngpus; ++i) {
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, i)!=cudaSuccess) return {};
        uuids[i] = props.uuid;
    }

    return uuids;
}

#endif

using uuid_range = std::pair<std::vector<uuid>::const_iterator,
                               std::vector<uuid>::const_iterator>;

std::ostream& operator<<(std::ostream& o, uuid_range rng) {
    o << "[";
    for (auto i=rng.first; i!=rng.second; ++i) {
        o << " " << int(i->bytes[0]);
    }
    return o << "]";
}


/*
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
*/

