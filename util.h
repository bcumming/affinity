#pragma once

#include <string>
#include <vector>

// Simple greedy algorithm that prints a sequence of integers as a set of
// ranges. Sorts the input, and removes duplicates.
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
std::string print_range(std::vector<int> v, int width = 3);

std::vector<std::string> consolidate(std::vector<std::vector<int>> affinities);
