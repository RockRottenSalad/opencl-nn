#pragma once

#include "kernels.hpp"
#include <CL/opencl.hpp>
#include<optional>

namespace lazyml {

namespace clwrapper {

    enum class SearchBy {
        VRAM,
        FREQ
    };

    #define FORWARD_METHOD(x) auto x() { return _kernels.x(_context, _device); }

    /**
    * Gets the best device based on search by parameter.
    * Will by default return device with highest amount of VRAM.
    * Returns empty optional if no device could be found.
    *
    * @param searchBy What devices should be compared for.
    * @return Device wrapped in optional or an empty optional if no device could be found.
    */
    std::optional<cl::Device> getBestDevice(SearchBy searchBy = SearchBy::VRAM);

    class clcontext {
        public:
        cl::Device _device;
        cl::Context _context;
        cl::CommandQueue _queue;


        clcontext(cl::Device device) : _device(device), _context({device}), _queue({_context, _device}) {}

        FORWARD_METHOD(get_vnn_kernels);
        FORWARD_METHOD(get_utils_kernels);

        private:
            kernels::kernelloader _kernels;
    };
}

}
