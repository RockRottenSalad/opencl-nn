
#include "model/vnn.hpp"
#include "utils.hpp"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/opencl.hpp>

#include <algorithm>

using namespace lazyml;
using namespace lazyml::models;

vnn::vnn(clwrapper::clcontext& con, std::vector<uint> &arch) 
: model(con) {

    const size_t n = arch.size();
    assert(n > 1);
    _layers = n;

    _neurons_per_layer = arch;

    // n = total number of layers where input is also counted as a layer

    // n-1, since there isn't a weight matrix or bias column vector for the inputs
    std::for_each(ALL(_biases_d), [n](auto &v){v.reserve(n-1);});
    std::for_each(ALL(_weights_d), [n](auto &v){v.reserve(n-1);});

    // n, since input is counted as a layer
    std::for_each(ALL(_activations_d), [n](auto &v){v.reserve(n);});

    // Add input activation column vector
    //this->add_matrix(_activations_d[MAIN_CL_BUFFERS], {1, arch[0]});
    //this->add_matrix(_activations_d[GRADIENT_CL_BUFFERS], {1, arch[0]});
    {
        size_t number_of_input_neurons = arch[0];
        this->add_matrix(_activations_d[MAIN_CL_BUFFERS], number_of_input_neurons);
        this->add_matrix(_activations_d[GRADIENT_CL_BUFFERS], number_of_input_neurons);
    }

    for(size_t i = 1; i < n; i++) {
        assert(arch[i] != 0 && "Neuron layer cannot have 0 neurons");

        // Add new weight matrix and randomize the initial weights
        // Number of rows corresponds to the number of columns in the previous activation column vector
        // Number of columns corresponds to the number of neurons in the current layer
        size_t rows = arch[i-1];
        size_t cols = arch[i];
        size_t n = rows * cols;
//        math::matrix w = {arch[i-1], arch[i]}; w.randomize();
        this->add_matrix(_weights_d[MAIN_CL_BUFFERS], n);
        this->add_matrix(_weights_d[GRADIENT_CL_BUFFERS], n);

        // Add new bias and randomize the initial biases
        // Bias is a column vector with size corresponding to number of neurons in the current layer
        //math::matrix b = {1, arch[i]}; b.randomize();
        cols = arch[i];
        this->add_matrix(_biases_d[MAIN_CL_BUFFERS], cols);
        this->add_matrix(_biases_d[GRADIENT_CL_BUFFERS], cols);

        // Activation is a column vector
        // Same dimensions as bias
//        math::matrix a = {1, arch[i]};
        cols = arch[i];
        this->add_matrix(_activations_d[MAIN_CL_BUFFERS], cols);
        this->add_matrix(_activations_d[GRADIENT_CL_BUFFERS], cols);
    }

    // Set the NDRange of the kernel to the width of the widest matrix
    // This is to ensure that there are enough threads to compute each matrix in parallel
    uint max_column = *std::max_element(ALL(arch));
    _kernel_range = cl::NDRange(max_column);

    _cost_kernel = _context.get_vnn_kernels().get().cost_kernel;
    _forward_kernel = _context.get_vnn_kernels().get().forward_kernel;

    _backprop_init_kernel = _context.get_vnn_kernels().get().backprop_init_kernel;
    _backprop_step_kernel = _context.get_vnn_kernels().get().backprop_step_kernel;

    _apply_gradient_kernel = _context.get_vnn_kernels().get().apply_gradient_kernel;
    _zero_kernel = _context.get_utils_kernels().get().zero;
    _copy_kernel = _context.get_utils_kernels().get().copy;

}

vnn::~vnn() {}

void vnn::run(clwrapper::memory<VNN_FLOAT_TYPE>& input, std::vector<VNN_FLOAT_TYPE> &output) {
    forward(input.get());

    size_t output_sz = static_cast<size_t>(_neurons_per_layer[_layers-1]);

    if(output.capacity() < output_sz) output.reserve(output_sz);
    while(output.size() < output_sz) output.emplace_back(0);

    // Read output from last activation layer
    _context._queue.enqueueReadBuffer(
        _activations_d[MAIN_CL_BUFFERS][_layers-1].get(), CL_TRUE, 0, sizeof(VNN_FLOAT_TYPE)*output_sz, output.data()
    );
}

std::vector<VNN_FLOAT_TYPE> vnn::run(clwrapper::memory<VNN_FLOAT_TYPE>& input) {
    size_t output_sz =  static_cast<size_t>(_neurons_per_layer[_layers-1]) ;
    std::vector<VNN_FLOAT_TYPE> output(output_sz);

    run(input, output);

    return output;
}


void vnn::train(
    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& input,
    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& output,
    uint iterations
) {
    assert(input.size() == output.size());

    size_t n = input.size();
    size_t input_sz = _neurons_per_layer[0];
    size_t output_sz = _neurons_per_layer[_layers-1];

    for(size_t i = 0; i < n; i++) {
        assert(input[i].size() == input_sz);
        assert(output[i].size() == output_sz);
    }


    for(uint epoch = 0; epoch < iterations; epoch++) {
        zero_gradient();

        for(size_t i = 0; i < n; i++) {
            zero_gradient_activations();
            forward(input[i].get());
            backprop(output[i].get());
        }

        apply_gradient( static_cast<cl_uint>(n) );
    }

    _context._queue.finish();
}

VNN_FLOAT_TYPE vnn::cost(
                std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& input,
                std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& expected_output
) {
    //assert(input.cols() == _neurons_per_layer[0] && output.cols() == _neurons_per_layer[_layers-1]);
    assert(input.size() == expected_output.size());

    size_t n = input.size();
    assert(n > 0);

    size_t input_sz = _neurons_per_layer[0];
    size_t output_sz = _neurons_per_layer[_layers-1];
    for(size_t i = 0; i < n; i++) {
        assert(input[i].size() == input_sz);
        assert(expected_output[i].size() == output_sz);
    }

    std::vector<VNN_FLOAT_TYPE> out(expected_output[0].size());
    VNN_FLOAT_TYPE err = 0.0f;

    for(size_t i = 0; i < n; i++) {
        run(input[i], out);

        for(size_t j = 0; j < out.size(); j++) {
            VNN_FLOAT_TYPE tmp = out[j] - expected_output[i][j];
            err += tmp*tmp;
        }
    }

    return err / static_cast<VNN_FLOAT_TYPE>(n);
}

void vnn::forward(cl::Buffer &input) {

    _copy_kernel.setArg(0, _activations_d[MAIN_CL_BUFFERS][0].get());
    _copy_kernel.setArg(1, input);
    _copy_kernel.setArg(2, sizeof(cl_uint), &_neurons_per_layer[0]);

    _context._queue.enqueueNDRangeKernel(_copy_kernel, cl::NullRange, _kernel_range);

    for(size_t i = 0; i < _layers-1; i++) {

        // arg[0] = weight matrix
        _forward_kernel.setArg(0, _weights_d[MAIN_CL_BUFFERS][i].get());
        // arg[1] = bias matrix
        _forward_kernel.setArg(1, _biases_d[MAIN_CL_BUFFERS][i].get());
        // arg[2] = activation matrix
        _forward_kernel.setArg(2, _activations_d[MAIN_CL_BUFFERS][i].get());

        // arg[3] = number of rows in weight matrix
        cl_uint rows = static_cast<cl_uint>(_neurons_per_layer[i]);
        _forward_kernel.setArg(3, sizeof(cl_uint), &rows);
        // arg[4] = number of columns in weight matrix
        cl_uint cols = static_cast<cl_uint>(_neurons_per_layer[i+1]);
        _forward_kernel.setArg(4, sizeof(cl_uint), &cols);

        _forward_kernel.setArg(5, _activations_d[MAIN_CL_BUFFERS][i+1].get());

        _context._queue.enqueueNDRangeKernel(_forward_kernel, cl::NullRange, _kernel_range);
    }

}

void vnn::backprop(cl::Buffer &output) {
    cl_uint n = _neurons_per_layer[_layers-1];
    _copy_kernel.setArg(0, _activations_d[GRADIENT_CL_BUFFERS][_layers-1].get());
    _copy_kernel.setArg(1, output);
    _copy_kernel.setArg(2, sizeof(cl_uint), &n);

    _context._queue.enqueueNDRangeKernel(_copy_kernel, cl::NullRange, _kernel_range);

    // Compute (aL - y)
    _backprop_init_kernel.setArg(0, _activations_d[MAIN_CL_BUFFERS][_layers-1].get());
    _backprop_init_kernel.setArg(1, _activations_d[GRADIENT_CL_BUFFERS][_layers-1].get());
    _backprop_init_kernel.setArg(2, sizeof(cl_uint), &n);
    _context._queue.enqueueNDRangeKernel(_backprop_init_kernel, cl::NullRange, _kernel_range);

    for(size_t l = _layers-1; l > 0; l--) {
        // weights matrix and weights gradient
        _backprop_step_kernel.setArg(0, _weights_d[MAIN_CL_BUFFERS][l-1].get());
        _backprop_step_kernel.setArg(1, _weights_d[GRADIENT_CL_BUFFERS][l-1].get());

        // Bias column vector
        _backprop_step_kernel.setArg(2, _biases_d[GRADIENT_CL_BUFFERS][l-1].get());

        // Activations matrix of current and previous layer
        _backprop_step_kernel.setArg(3, _activations_d[MAIN_CL_BUFFERS][l].get());
        _backprop_step_kernel.setArg(4, _activations_d[MAIN_CL_BUFFERS][l-1].get());

        // Buffers used for intermediary values
        _backprop_step_kernel.setArg(5, _activations_d[GRADIENT_CL_BUFFERS][l].get());
        _backprop_step_kernel.setArg(6, _activations_d[GRADIENT_CL_BUFFERS][l-1].get());

        // Dimensions of weight matrix
        cl_uint cols = _neurons_per_layer[l];
        cl_uint rows = _neurons_per_layer[l-1];
        _backprop_step_kernel.setArg(7, sizeof(cl_uint), &cols);
        _backprop_step_kernel.setArg(8, sizeof(cl_uint), &rows);

        _context._queue.enqueueNDRangeKernel(_backprop_step_kernel, cl::NullRange, _kernel_range);
    }
}

void vnn::apply_gradient(cl_uint n) {

    _apply_gradient_kernel.setArg(6, sizeof(cl_uint), &n);

    for(size_t l = 0; l < _layers-1; l++) {
        _apply_gradient_kernel.setArg(0, _weights_d[MAIN_CL_BUFFERS][l].get());
        _apply_gradient_kernel.setArg(1, _weights_d[GRADIENT_CL_BUFFERS][l].get());

        _apply_gradient_kernel.setArg(2, _biases_d[MAIN_CL_BUFFERS][l].get());
        _apply_gradient_kernel.setArg(3, _biases_d[GRADIENT_CL_BUFFERS][l].get());

        cl_uint cols = _neurons_per_layer[l+1];
        cl_uint rows = _neurons_per_layer[l];

        _apply_gradient_kernel.setArg(4, sizeof(cl_uint), &cols);
        _apply_gradient_kernel.setArg(5, sizeof(cl_uint), &rows);

        _context._queue.enqueueNDRangeKernel(_apply_gradient_kernel, cl::NullRange, _kernel_range);
    }
}

void vnn::zero_gradient_activations() {
    for(size_t l = 0; l < _layers; l++) {
        cl_uint cols = _neurons_per_layer[l];

        _zero_kernel.setArg(0, _activations_d[GRADIENT_CL_BUFFERS][l].get());
        _zero_kernel.setArg(1, sizeof(cl_uint), &cols);
        _context._queue.enqueueNDRangeKernel(_zero_kernel, cl::NullRange, _kernel_range);
    }
}

void vnn::zero_gradient() {
    for(size_t l = 0; l < _layers-1; l++) {
        cl_uint rows = _neurons_per_layer[l];
        cl_uint cols = _neurons_per_layer[l+1];
        cl_uint n = rows * cols;

        _zero_kernel.setArg(0, _weights_d[GRADIENT_CL_BUFFERS][l].get());
        _zero_kernel.setArg(1, sizeof(cl_uint), &n);
        _context._queue.enqueueNDRangeKernel(_zero_kernel, cl::NullRange, _kernel_range);

        _zero_kernel.setArg(0, _biases_d[GRADIENT_CL_BUFFERS][l].get());
        _zero_kernel.setArg(1, sizeof(cl_uint), &cols);
        _context._queue.enqueueNDRangeKernel(_zero_kernel, cl::NullRange, _kernel_range);
    }
}


void vnn::add_matrix(std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> &matrix_list, size_t n) {
    size_t index = matrix_list.size();

    bool shouldRandomizeValues = true;
    matrix_list.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(_context, shouldRandomizeValues, n)  );

    bool shouldBlock = false;
    matrix_list[index].writeToDevice(shouldBlock);
}
