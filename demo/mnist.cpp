// Very ugly and slow
// Maytbe it's time for stochastic descent

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <bit>
#include <memory>
#include <numeric>
#include <span>

#include "clwrapper.hpp"
#include "lazyml.hpp"

using namespace lazyml;

// The images are 2D, but stored in a list making the data 3D
#define DIMENSIONS 3
// All images are 28x28
#define SIZE 28
#define PIXELS_PER_IMAGE SIZE*SIZE
// 60k images in the mnist dataset
#define ENTRIES 60000
// Byte value to indicate that the data is made up of unsigned bytes
#define EXPECTED_DATA 0x08

typedef char byte;
typedef unsigned short uint8;

// The mnist file uses Big Endian byte order, the computer this is running on might not
#define read_int(FILESTREAM, OUTPUT)\
    FILESTREAM.read((char*)&OUTPUT, sizeof(uint));\
    if constexpr (std::endian::native == std::endian::little)\
    OUTPUT = __builtin_bswap32(OUTPUT)


std::unique_ptr<VNN_FLOAT_TYPE> read_mnist(const std::string &filepath) {
    std::ifstream input_file(filepath, std::ios_base::binary);
    assert(input_file.is_open());

    // First two bytes in magic number are always zero and should be ignored
    input_file.ignore(2);

    byte data_type, dimensions;
    input_file.read(&data_type, 1);
    input_file.read(&dimensions, 1);

    std::cout << "data_type: " << (int)data_type << ", dim: " << (int)dimensions << std::endl;

    assert(data_type == EXPECTED_DATA);
    //assert(data_type == EXPECTED_DATA && dimensions == DIMENSIONS);

    std::vector<uint8> dimensionsSize; dimensionsSize.reserve(dimensions);
    uint buffer;
    for(size_t i = 0; i < dimensions; i++) {
        read_int(input_file, buffer);
        dimensionsSize.emplace_back(buffer);
    }

    size_t n = std::accumulate(ALL(dimensionsSize), 1, [](size_t acc, uint x){ return acc * static_cast<size_t>(x); });

    if(dimensionsSize.size() == 1)
        assert(n == ENTRIES);
    else 
        assert(n == ENTRIES * SIZE * SIZE);
    //assert(y == SIZE && x == SIZE && entries == ENTRIES);

    VNN_FLOAT_TYPE *data = new VNN_FLOAT_TYPE[n];

    byte byte_buffer;
    for(size_t i = 0; i < n; i++) {
        input_file.read(&byte_buffer, 1);
        data[i] = static_cast<VNN_FLOAT_TYPE>(byte_buffer);

    }

    return std::unique_ptr<VNN_FLOAT_TYPE>(data);
}


int main() {
    srand(time(nullptr)); 
    

    std::unique_ptr<VNN_FLOAT_TYPE> input_data = read_mnist("data/train-images-idx3-ubyte");
    std::unique_ptr<VNN_FLOAT_TYPE> output_data = read_mnist("data/train-labels-idx1-ubyte");

    cl::Device default_device = utils::value_or_panic(clwrapper::getBestDevice(), "Could not any find device");
    clwrapper::clcontext con = {default_device};

    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> inputs; inputs.reserve(ENTRIES);
    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> outputs; inputs.reserve(ENTRIES);

    for(size_t i = 0; i < ENTRIES * PIXELS_PER_IMAGE; i += PIXELS_PER_IMAGE) {
        std::span<VNN_FLOAT_TYPE> span = {input_data.get() + i, input_data.get() + i + PIXELS_PER_IMAGE};
        inputs.emplace_back(clwrapper::memory<VNN_FLOAT_TYPE>(con, span));
    }

    std::vector<VNN_FLOAT_TYPE> output_buffer(10, 0);
    for(size_t i = 0; i < ENTRIES; i++) {
        size_t x = static_cast<size_t>(output_data.get()[i]);

        std::fill(ALL(output_buffer), 0);
        output_buffer[x] = 1.0f;

        outputs.emplace_back(clwrapper::memory<VNN_FLOAT_TYPE>(con, output_buffer));
        
        const bool shouldBlock = false;
        inputs[i].writeToDevice(shouldBlock);
        outputs[i].writeToDevice(shouldBlock);
    }

    std::cout << "inputs len: " << inputs.size() << std::endl;
    std::cout << "outputs len: " << outputs.size() << std::endl;

    // 784 input neurons(28*28) for the image
    // two hidden layers with 16 neurons each
    // 10 outputs neurons, one for each possible digit[0-9]
    std::vector<cl_uint> arch = {784, 16, 16, 10};

    models::vnn nn {con, arch};

    std::cout << "COST: " << nn.cost(inputs, outputs) << std::endl;
    nn.train(inputs, outputs, 500, 20.0);
    std::cout << "COST: " << nn.cost(inputs, outputs) << std::endl;



    return 0;
}

