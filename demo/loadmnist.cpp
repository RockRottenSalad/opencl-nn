// Very ugly and slow
// Maytbe it's time for stochastic descent
#include <iostream>
#include <cassert>

#include "clwrapper.hpp"

#define ENTRIES 10000
#include "mnistdata.hpp"

using namespace lazyml;

int main() {
    srand(time(nullptr)); 
    
    cl::Device default_device = utils::value_or_panic(clwrapper::getBestDevice(), "Could not any find device");
    clwrapper::clcontext con = {default_device};

    auto inputs_outputs  = get_mnist_data(
        con,
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );

    auto inputs = inputs_outputs.first;
    auto outputs = inputs_outputs.second;

    std::cout << "inputs len: " << inputs.size() << std::endl;
    std::cout << "outputs len: " << outputs.size() << std::endl;

    std::cout << "Image\n";
    for(size_t i = 0; i < PIXELS_PER_IMAGE; i++) {
        std::cout << inputs[0][i] << " ";
    }
    std::cout << std::endl;

    // 784 input neurons(28*28) for the image
    // two hidden layers with 16 neurons each
    // 10 outputs neurons, one for each possible digit[0-9]
    std::vector<cl_uint> arch = {784, 16, 16, 10};

    models::vnn nn {con, "mnist2.nn"};

    float c0 = nn.cost(inputs, outputs);
    std::cout << "COST: " << c0 << std::endl;

    for(size_t i = 0; i < inputs.size(); i++) {
        auto result = nn.run(inputs[i]);

        std::cout << "Result: ";
        for(size_t j = 0; j < 10; j++) {
            std::cout << result[j] << " ";
        }
        std::cout << "\nExpected: ";
        for(size_t j = 0; j < 10; j++) {
            std::cout << outputs[i][j] << " ";
        }
        std::cout << std::endl;

    }

//    nn.train(inputs, outputs, 3, 20.0);
//
//    float c1 = nn.cost(inputs, outputs);
//    std::cout << "COST: " << c1 << std::endl;

//    if(c1 < c0) nn.serialize("mnist.nn");

    return 0;
}

