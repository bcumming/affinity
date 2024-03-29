#include <array>
#include <ostream>
#include <vector>

#include <cuda_runtime.h>

// Store cudaUUID_t in a byte array for easy type punning and comparison.
// 128 bit uuids are not just for GPUs: they are used in many applications, so
// we call the type uuid, instead of a gpu-specific name.
// uuids follow the most common storage format, which is big-endian.
struct alignas(8) uuid {
    std::array<unsigned char, 16> bytes;

    uuid() {
        std::fill(bytes.begin(), bytes.end(), 0);
    }

    uuid(const uuid&) = default;

    uuid(cudaUUID_t i) {
        const unsigned char* b = reinterpret_cast<const unsigned char*>(&i);
        std::copy(b, b+sizeof(cudaUUID_t), bytes.begin());
    }
};

// Test GPU uids for equality
bool operator==(const uuid& lhs, const uuid& rhs);

// Strict lexographical ordering of GPU uids
bool operator<(const uuid& lhs, const uuid& rhs);

// Print a uuid
// Prints in "standard" uuid format of lower case hexadecimal, with hyphens
// in the 8-4-4-4-12 pattern:
//      xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
std::ostream& operator<<(std::ostream& o, const uuid& id);

// Returns a vector of of the unique gpu identifiers for all
// gpus available to this process.
std::vector<uuid> get_gpu_uuids();

