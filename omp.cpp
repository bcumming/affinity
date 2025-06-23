#include <iomanip>
#include <iostream>
#include <sstream>

#include <omp.h>

#include "affinity.h"
#include "util.h"

int main(void) {

    auto num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> cores(num_threads);

#pragma omp parallel
    {
        cores[omp_get_thread_num()] = get_affinity();
    }

    const auto strings = consolidate(cores);

    /*
    for (auto i = 0; i < num_threads; ++i) {
        std::cout << "thread " << std::setw(3) << i << " on cores ["
                  << strings[i] << "]" << std::endl;
    }
    */
    for (auto& s : strings) {
        std::cout << s << std::endl;
    }
}
