#include <iomanip>
#include <iostream>
#include <sstream>

#include <omp.h>

#include "affinity.h"
#include "util.h"

int main(void) {

    auto num_threads = omp_get_max_threads();
    std::vector<std::string> strings(num_threads);

    #pragma omp parallel
    {
        strings[omp_get_thread_num()] = print_as_ranges(get_affinity());
    }

    for(auto i=0; i<num_threads; ++i) {
        std::cout << "thread " << std::setw(3) << i
                  << " on cores [" << strings[i] << "]"
                  << std::endl;
    }
}
