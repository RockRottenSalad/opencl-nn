#pragma once

#include "clwrapper.hpp"
#include "model.hpp"
#include "math/math.hpp"
#include "utils.hpp"
#include <CL/opencl.hpp>
#include <algorithm>

#ifndef VNN_FLOAT_TYPE
#define VNN_FLOAT_TYPE float
#endif

namespace lazyml {

namespace models {

#define MAIN_CL_BUFFERS 0
#define GRADIENT_CL_BUFFERS 1

    class vnn : model<VNN_FLOAT_TYPE> {
        public:
        vnn(clwrapper::clcontext& con, std::vector<uint> &arch);
        ~vnn();


        std::vector<VNN_FLOAT_TYPE> run(clwrapper::memory<VNN_FLOAT_TYPE>& input);
        void run(clwrapper::memory<VNN_FLOAT_TYPE>& input, std::vector<VNN_FLOAT_TYPE> &output);

        void train(
                std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& input,
                std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& output,
                uint iterations,
                VNN_FLOAT_TYPE learning_rate
        );
        VNN_FLOAT_TYPE cost(
                std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& input,
                std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& output
        );

        void serialize(const std::string &filename);

        private:
        std::vector<cl_uint> _neurons_per_layer;
        size_t _layers;

        // Index 0 = Actual weights, biases and activations(counting input, intermediate and output as activations)
        // Index 1 = Gradient. Activations gradient is used as buffers for some certain calculations in backpropagation
        std::array<std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>, 2> _weights_d, _biases_d, _activations_d;

        // Maybe use singleton pattern for this and only instantiate if get function is called?
        // Multiple instances of vnn can rely on same program. There might be delay though 
        cl::Program _program;

        cl::Kernel _cost_kernel;
        cl::Kernel _forward_kernel;
        cl::Kernel _backprop_init_kernel, _backprop_step_kernel, _apply_gradient_kernel, _zero_kernel, _copy_kernel;

        cl::NDRange _kernel_range;

        void forward(cl::Buffer &input);
        void backprop(cl::Buffer &output);

        void apply_gradient(cl_uint n, cl_float learning_rate);
        void zero_gradient_activations();
        void zero_gradient();

        void add_matrix(std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> &matrix_list, size_t n);
        void read_from_device();

    };

}

}

