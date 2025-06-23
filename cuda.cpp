#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <vector>

#include <cuda_runtime.h>

#include "cuda.hpp"

#include <iostream>

// Test GPU uids for equality
bool operator==(const uuid& lhs, const uuid& rhs) {
    for (auto i = 0u; i < lhs.bytes.size(); ++i) {
        if (lhs.bytes[i] != rhs.bytes[i])
            return false;
    }
    return true;
}

// Strict lexographical ordering of GPU uids
bool operator<(const uuid& lhs, const uuid& rhs) {
    for (auto i = 0u; i < lhs.bytes.size(); ++i) {
        if (lhs.bytes[i] < rhs.bytes[i])
            return true;
        if (lhs.bytes[i] > lhs.bytes[i])
            return false;
    }
    return false;
}

std::ostream& operator<<(std::ostream& o, const uuid& id) {
    o << std::hex << std::setfill('0');

    bool first = true;
    int ranges[6] = {0, 4, 6, 8, 10, 16};
    for (int i = 0; i < 5;
         ++i) { // because std::size isn't available till C++17
        if (!first)
            o << "-";
        for (auto j = ranges[i]; j < ranges[i + 1]; ++j) {
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
//      CUDA_VISIBLE_DEVICES="1, 0, hello" -> {1,0}
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
    for (auto& s : strings) {
        try {
            int v = std::stoi(s);
            if (v < 0 || v >= ngpu)
                break;
            values.push_back(v);
        } catch (std::exception e) {
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
    auto pos =
        std::remove_if(str, str + n, [](char c) { return !std::isxdigit(c); });

    // Converts a single hex character, i.e. 0123456789abcdef, to int
    // Assumes that input is a valid hex character.
    auto hex_c2i = [](char c) -> unsigned char {
        return std::isalpha(c) ? c - 'a' + 10 : c - '0';
    };

    // Convert pairs of characters into single bytes.
    for (int i = 0; i < 16; ++i) {
        const char* s = str + 2 * i;
        result.bytes[i] = (hex_c2i(s[0]) << 4) + hex_c2i(s[1]);
    }

    return result;
}

std::vector<uuid> get_gpu_uuids() {
    // get number of devices
    int ngpus = 0;
    if (cudaGetDeviceCount(&ngpus) != cudaSuccess)
        return {};

    // store the uuids
    std::vector<uuid> uuids(ngpus);
    // using CUDA 10 or later, determining uuid of GPUs is easy!
    for (int i = 0; i < ngpus; ++i) {
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, i) != cudaSuccess)
            return {};
        uuids[i] = props.uuid;
    }

    return uuids;
}

using uuid_range = std::pair<std::vector<uuid>::const_iterator,
                             std::vector<uuid>::const_iterator>;

std::ostream& operator<<(std::ostream& o, uuid_range rng) {
    o << "[";
    for (auto i = rng.first; i != rng.second; ++i) {
        o << " " << int(i->bytes[0]);
    }
    return o << "]";
}
