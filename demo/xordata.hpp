#ifndef XORDATA_HPP
#define XORDATA_HPP

#include "clwrapper.hpp"
#include "lazyml.hpp"
#include<vector>

using namespace lazyml;

std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> data_input(clwrapper::clcontext &con) {

    // 4 possible inputs
    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> inputs;
    inputs.reserve(4);

    // {0,0}, {0,1}, {1,0}, {1,1}
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0,0}) );
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0,1}) );
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1,0}) );
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1,1}) );
    // Write the data to VRAM                                 v Don't block
    std::for_each(ALL(inputs), [](auto &x) { x.write_to_device(false);  });

    return inputs;
}

std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> data_output(clwrapper::clcontext &con) {

    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> outputs;
    outputs.reserve(4);

    // The 4 outputs for each input: {0,0} => 0, {0,1} => 1, {1,0} => 1, {1,1} => 0
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0}) );
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1}) );
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1}) );
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0}) );
    std::for_each(ALL(outputs), [](auto &x) { x.write_to_device(false);  });

    return outputs;
}

#endif
