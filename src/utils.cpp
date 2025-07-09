
#include "utils.hpp"

#include<vector>
#include<fstream>
#include<iostream>

using namespace lazyml;

uint utils::nearest_power_of_two(uint x) {
    uint y = 1;
    while(y < x) y = y << 1;
    return y;
}

std::string utils::file_to_string(std::string filepath) {
    std::ifstream ifs(filepath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

    std::ifstream::pos_type fileSize = ifs.tellg();
    if (fileSize < 0)
        return std::string();

    ifs.seekg(0, std::ios::beg);

    std::vector<char> bytes(fileSize);
    ifs.read(&bytes[0], fileSize);

    return std::string(&bytes[0], fileSize);
}


