#pragma once

#include <algorithm>
#include <sstream>
#include <vector>

// Simple greedy algorithm that prints a sequence of integers as a set of ranges.
// Sorts the input, and removes duplicates.
//
// Some example output:
// {42}
// 42
//
// {0,1}
// 0, 1
//
// {0,1,2,3}
// 0:3
//
// {0,1,2,3, 8,9,10,11}
// 0:3, 8:11
//
// {0,2,4,6, 21,22,23, 30,35,40}
// 0:2:6, 21:23, 30:5:40
//
// {0,2,4, 8, 21,22,23}
// 0:2:4, 8, 21:23
//
// {0,1,7,12}
// 0, 1, 7, 12
//
// {0, 1, 1, 13, 12, 3, 2, 2}
// 0:3, 12, 13

std::string print_as_ranges(std::vector<int> v) {
    std::stringstream s;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    if (v.size()==0) {
        return "";
    }

    bool first = true;
    auto it = v.begin();
    while (it != v.end()) {
        if (!first) {
            s << ", ";
        }
        first = false;
        auto pos = it; 
        s << *it;
        int delta = *(it+1)-(*it);
        while ((it+1)!=v.end() && *(it+1)-(*it) == delta) {
            ++it;
        }
        auto dist = std::distance(pos, it);
        if (dist>1u) {
            if (delta>1) {
                s << ":" << delta << ":" << *it;
            }
            else {
                s << ":" << *it;
            }
        }

        if (dist!=1u) {
            ++it;
        }
    }

    return s.str();
}

