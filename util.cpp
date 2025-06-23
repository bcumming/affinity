#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

#include "util.h"

// sort the input vector and remove duplicates
// modifies the input
template <typename T> void deduplicate(std::vector<T>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

// INPUT: a sorted vector of integers
// OUTPUT: a short string representation of the input range
std::string print_range(std::vector<int> v, int width) {
    std::stringstream s;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    if (v.size() == 0) {
        return "";
    }
    if (v.size() > 1) {
        s << "[";
    }

    bool first = true;
    auto it = v.begin();
    while (it != v.end()) {
        if (!first) {
            s << ",";
        }
        first = false;
        auto pos = it;
        s << std::setw(width) << *it;
        int delta = *(it + 1) - (*it);
        while ((it + 1) != v.end() && *(it + 1) - (*it) == delta) {
            ++it;
        }
        auto dist = std::distance(pos, it);
        if (dist > 1u) {
            if (delta > 1) {
                s << ":" << delta << ":" << std::setw(width) << *it;
            } else {
                s << ":" << std::setw(width) << *it;
            }
        }

        if (dist != 1u) {
            ++it;
        }
    }
    if (v.size() > 1) {
        s << "]";
    }

    return s.str();
}

// for:
//   sorting vectors of vectors
//   searching vectors of sorted vectors
bool operator<(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    auto n = std::min(lhs.size(), rhs.size());
    for (auto i = 0u; i < n; ++i) {
        if (lhs[i] < rhs[i]) {
            return true;
        }
        if (rhs[i] < lhs[i]) {
            return false;
        }
    }
    // the two arrays were equal up to entry n
    if (lhs.size() < rhs.size()) {
        return true;
    }

    return false;
}

// INPUT: a vector of core affinities, one per thread
// OUTPUT: a list of strings that describe the groupings of threads and CPU
// cores
std::vector<std::string> consolidate(std::vector<std::vector<int>> affinities) {
    // for each affinity set, sort and deduplicate the affinities
    // this ensure that all affinity sets are canonicalised
    for (auto& v : affinities) {
        deduplicate(v);
    }

    // If there are different threads with affinity with the same set of cores,
    // we want to group these together. The first step is to determine the set
    // of unique "coresets" that in the list of affinities. Sort and deduplicate
    // the affinities and store this in coresets.
    auto coresets = affinities;
    std::sort(coresets.begin(), coresets.end());
    deduplicate(coresets);

    const auto nsets = coresets.size();
    const auto ntids = affinities.size();

    const int twidth = ntids < 10 ? 1 : ntids < 100 ? 2 : 3;

    // For each thread, find the unique coreset that it has affinity with.
    // threadsets[tid] will contain a sorted list of threads that have affinity
    // with thread tid.
    auto threadsets = std::vector<std::vector<int>>{nsets};
    for (auto tid = 0u; tid < ntids; ++tid) {
        const auto pos =
            std::lower_bound(coresets.begin(), coresets.end(), affinities[tid]);
        const auto idx = std::distance(coresets.begin(), pos);
        threadsets[idx].push_back(tid);
    }

    // generate the output strings
    auto result = std::vector<std::string>{};
    result.reserve(nsets);
    for (auto i = 0u; i < nsets; ++i) {
        std::stringstream s;
        const auto& threads = threadsets[i];
        const auto& cores = coresets[i];
        s << (threads.size() > 1u ? "threads " : "thread ")
          << print_range(threads, twidth)
          << (threads.size() > 1u ? " -> " : " -> ")
          << (cores.size() > 1u ? "cores " : "core ") << print_range(cores);
        result.push_back(s.str());
    }

    return result;
}
