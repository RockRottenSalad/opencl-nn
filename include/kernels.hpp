#pragma once

#include <CL/opencl.hpp>
#include <optional>


#define KERNEL_VNN_SOURCE_PATH "cl/vanilla_nn_kernel.cl"
#define KERNEL_UTILS_SOURCE_PATH "cl/utils.cl"

namespace lazyml {

    namespace kernels {

        struct vnn_kernels {
            cl::Program program;

            cl::Kernel cost_kernel,
                       forward_kernel,
                       backprop_init_kernel,
                       backprop_step_kernel,
                       apply_gradient_kernel;
        };

        struct utils_kernels {
            cl::Program program;
            cl::Kernel rand, zero;
        };

        class kernelloader {
            private:
                std::optional<vnn_kernels> _vnn;
                std::optional<utils_kernels> _utils;
                static void compile(cl::Program program, cl::Device device);

            public:
                kernelloader();

                 std::reference_wrapper<vnn_kernels> get_vnn_kernels(cl::Context context, cl::Device device);
                std::reference_wrapper<utils_kernels> get_utils_kernels(cl::Context context, cl::Device device);

        };



    }

}

