
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>

#include "lazyml.hpp"
#include "xordata.hpp"

using namespace lazyml;

bool file_exists(const std::string &filename) {
    std::ifstream in(filename, std::ios::in);

    return in.is_open();
}

int main() {
    srand(time(nullptr)); 

    const std::string file = "xor.nn";
    if(!file_exists(file)) {
        std::cout << "'xor.nn' not found, xor 'xor' target first to generate serialized model" << std::endl;
        return 0;
    }

    // Find best device(GPU)
    cl::Device default_device = utils::value_or_panic(clwrapper::getBestDevice(), "Could not any find device");

    // Create context using previously found GPU
    clwrapper::clcontext con = {default_device};

    // Load the serialized xor model
    models::vnn nn {con, "xor.nn"};

    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> inputs = data_input(con);
    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> outputs = data_output(con);

    // Print out the current cost
    std::cout << "COST: " << nn.cost(inputs, outputs) << "\n";

    // Current output for each input. This will be absolute garbage
    std::cout << "0 0 = " << (nn.run(inputs[0]))[0] << "\n";
    std::cout << "0 1 = " << (nn.run(inputs[1]))[0] << "\n";
    std::cout << "1 0 = " << (nn.run(inputs[2]))[0] << "\n";
    std::cout << "1 1 = " << (nn.run(inputs[3]))[0] << "\n";

    return 0;
}

