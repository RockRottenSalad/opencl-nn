
#include "kernels.hpp"

#include "utils.hpp"
#include <functional>
#include <optional>
#include <cassert>
#include <unistd.h>

using namespace lazyml;
using namespace kernels;

kernelloader::kernelloader() 
:   _vnn(std::nullopt),
    _utils(std::nullopt)
{}

std::reference_wrapper<utils_kernels> kernelloader::get_utils_kernels(cl::Context context, cl::Device device) {
    if(!_utils.has_value()) {
        utils_kernels new_kernels = {};

        std::string source = utils::file_to_string(KERNEL_UTILS_SOURCE_PATH);
        assert(source.size() != 0 && "Could not find source");

        new_kernels.program = cl::Program(context, source);

        compile(new_kernels.program, device);

        // ---
        new_kernels.zero = cl::Kernel(new_kernels.program, "zero");
        new_kernels.copy = cl::Kernel(new_kernels.program, "copy");
        new_kernels.rand = cl::Kernel(new_kernels.program, "rand_buffer");

        _utils = new_kernels;
    }

    return _utils.value();
}

std::reference_wrapper<vnn_kernels> kernelloader::get_vnn_kernels(cl::Context context, cl::Device device) {
    if(!_vnn.has_value()) {
        vnn_kernels new_kernels = {};

        std::string source = utils::file_to_string(KERNEL_VNN_SOURCE_PATH);
        assert(source.size() != 0 && "Could not find source");

        new_kernels.program = cl::Program(context, source);

        compile(new_kernels.program, device);

        // ---
        new_kernels.forward_kernel = cl::Kernel(new_kernels.program, "forward");
        new_kernels.cost_kernel = cl::Kernel(new_kernels.program, "cost");

        // Backprop kernels
        new_kernels.backprop_init_kernel = cl::Kernel(new_kernels.program, "backprop_delta_init");
        new_kernels.backprop_step_kernel = cl::Kernel(new_kernels.program, "backprop_step");
        new_kernels.apply_gradient_kernel = cl::Kernel(new_kernels.program, "apply_gradient");

        _vnn = new_kernels;
    }

    return _vnn.value();
}

void kernelloader::compile(cl::Program program, cl::Device device) {
    int status = program.build(device);

    if(status != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        std::exit(-1);
    }

}

