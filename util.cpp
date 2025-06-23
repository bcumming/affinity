#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "util.h"

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

bool operator==(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (auto i = 0u; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

// sort the input vector and remove duplicates
template <typename T> void deduplicate(std::vector<T>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

std::string print_range(std::vector<int> v) {
    std::stringstream s;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    if (v.size() == 0) {
        return "";
    }

    bool first = true;
    auto it = v.begin();
    while (it != v.end()) {
        if (!first) {
            s << ",";
        }
        first = false;
        auto pos = it;
        s << *it;
        int delta = *(it + 1) - (*it);
        while ((it + 1) != v.end() && *(it + 1) - (*it) == delta) {
            ++it;
        }
        auto dist = std::distance(pos, it);
        if (dist > 1u) {
            if (delta > 1) {
                s << "-" << delta << "-" << *it;
            } else {
                s << "-" << *it;
            }
        }

        if (dist != 1u) {
            ++it;
        }
    }

    return s.str();
}

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& input) {
    bool first = true;
    for (auto& x : input) {
        o << (first ? "" : " ") << x;
        first = false;
    }
    return o;
}

template <typename T>
std::ostream& operator<<(std::ostream& o,
                         const std::vector<std::vector<T>>& input) {
    for (auto& x : input) {
        o << x << "\n";
    }
    return o;
}

// INPUT: a vector of affinities, one per thread
std::vector<std::string> consolidate(std::vector<std::vector<int>> affinities) {
    for (auto& v : affinities) {
        deduplicate(v);
    }
    // std::cout << "===== input affinities\n" << affinities << "\n";

    auto coresets = affinities;
    std::sort(coresets.begin(), coresets.end());
    deduplicate(coresets);

    // std::cout << "===== input coresets\n" << coresets << "\n";

    const auto nsets = coresets.size();
    const auto ntids = affinities.size();

    auto threadsets = std::vector<std::vector<int>>{nsets};
    for (auto tid = 0u; tid < ntids; ++tid) {
        const auto pos =
            std::lower_bound(coresets.begin(), coresets.end(), affinities[tid]);
        const auto idx = std::distance(coresets.begin(), pos);
        threadsets[idx].push_back(tid);
    }

    // std::cout << "===== input threadsets\n" << threadsets << "\n";

    auto result = std::vector<std::string>{};
    result.reserve(nsets);
    for (auto i = 0u; i < nsets; ++i) {
        std::stringstream s;
        const auto& threads = threadsets[i];
        const auto& cores = coresets[i];
        s << (threads.size() > 1u ? "threads [" : "thread ")
          << print_range(threads) << (threads.size() > 1u ? "] -> " : " -> ")
          << (cores.size() > 1u ? "cores [" : "core ") << print_range(cores)
          << (cores.size() > 1u ? "}" : "");
        result.push_back(s.str());
    }

    // std::cout << "================== messages\n" << result << "\n";

    return result;
}

std::string print_as_ranges(std::vector<int> v) {
    std::stringstream s;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    if (v.size() == 0) {
        return "";
    }

    bool first = true;
    auto it = v.begin();
    while (it != v.end()) {
        if (!first) {
            s << ",";
        }
        first = false;
        auto pos = it;
        s << *it;
        int delta = *(it + 1) - (*it);
        while ((it + 1) != v.end() && *(it + 1) - (*it) == delta) {
            ++it;
        }
        auto dist = std::distance(pos, it);
        if (dist > 1u) {
            if (delta > 1) {
                s << "-" << delta << "-" << *it;
            } else {
                s << "-" << *it;
            }
        }

        if (dist != 1u) {
            ++it;
        }
    }

    return s.str();
}
