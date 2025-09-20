
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "lazyml.hpp"

using namespace lazyml;

int main() {
    srand(time(nullptr)); 

    // Find best device(GPU)
    cl::Device default_device = utils::value_or_panic(clwrapper::getBestDevice(), "Could not any find device");

    // Create context using previously found GPU
    clwrapper::clcontext con = {default_device};

    // Just need to make this less hideous
    //
    // The 4 different inputs for the XOR gate: {0,0},{0,1},{1,0},{1,1}
    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> inputs;
    inputs.reserve(4);
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0,0}) );
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0,1}) );
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1,0}) );
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1,1}) );
    // Write the data to VRAM                                 v Don't block
    std::for_each(ALL(inputs), [](auto &x) { x.writeToDevice(false);  });

    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> outputs;
    outputs.reserve(4);

    // The 4 outputs for each input: {0,0} => 0, {0,1} => 1, {1,0} => 1, {1,1} => 0
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0}) );
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1}) );
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1}) );
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0}) );
    std::for_each(ALL(outputs), [](auto &x) { x.writeToDevice(false);  });

    // Neural network with:
    // 2 input neurons.
    // A hidden layer with 2 neurons
    // A single output neuron.
    std::vector<cl_uint> arch = {2, 2, 1};
    models::vnn nn {con, arch};

    // Print out the current cost
    std::cout << "COST: " << nn.cost(inputs, outputs) << "\n";

    // Current output for each input. This will be absolute garbage
    std::cout << "0 0 = " << (nn.run(inputs[0]))[0] << "\n";
    std::cout << "0 1 = " << (nn.run(inputs[1]))[0] << "\n";
    std::cout << "1 0 = " << (nn.run(inputs[2]))[0] << "\n";
    std::cout << "1 1 = " << (nn.run(inputs[3]))[0] << "\n";

    // Train the model for 10000 iterations
    nn.train(inputs, outputs, 10000);

    // Print out the new cost after training
    std::cout << "COST: " << nn.cost(inputs, outputs) << std::endl;

    // The output for each input after training
    std::cout << "0 0 = " << (nn.run(inputs[0]))[0] << std::endl;
    std::cout << "0 1 = " << (nn.run(inputs[1]))[0] << std::endl;
    std::cout << "1 0 = " << (nn.run(inputs[2]))[0] << std::endl;
    std::cout << "1 1 = " << (nn.run(inputs[3]))[0] << std::endl;
}

