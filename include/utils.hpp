#pragma once

#include<string>
#include<optional>
#include<iostream>

#define ALL(container) (container).begin(), (container).end()

namespace lazyml {

namespace utils {

    // Given equation x = 2^y
    // Returns ceil(y)
    //
    // Used to find ideal work group size for kernels
    uint nearest_power_of_two(uint x);

    std::string file_to_string(std::string filepath);

    template<typename T>
    T value_or_panic(const std::optional<T>& opt, const std::string& msg) {
        if(opt) {
            return opt.value();
        }

        std::cout << msg << std::endl;
        exit(-1);
    }

}
}

