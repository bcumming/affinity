#include <algorithm>
#include <array>
#include <ostream>
#include <vector>

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

// Strict lexographical ordering of GPU uids
bool operator<(const uuid& lhs, const uuid& rhs);

std::ostream& operator<<(std::ostream& o, const uuid& id);


std::vector<uuid> get_gpu_uuids();

