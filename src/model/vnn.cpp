
#include "model/vnn.hpp"
#include "utils.hpp"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/opencl.hpp>

#include <algorithm>
#include <fstream>
#include <cstdint>

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

    bool shouldRandomize = true;

    // Add input activation column vector
    this->add_matrix_pairs(_activations_d, shouldRandomize, _neurons_per_layer[0]);

    for(size_t i = 1; i < n; i++) {
        assert(arch[i] != 0 && "Neuron layer cannot have 0 neurons");

        // Add new weight matrix and randomize the initial weights
        // Number of rows corresponds to the number of columns in the previous activation column vector
        // Number of columns corresponds to the number of neurons in the current layer
        size_t rows = arch[i-1];
        size_t cols = arch[i];
        size_t n = rows * cols;
        this->add_matrix_pairs(_weights_d, n, shouldRandomize);

        // Add new bias and randomize the initial biases
        // Bias is a column vector with size corresponding to number of neurons in the current layer
        this->add_matrix_pairs(_biases_d, cols, shouldRandomize);

        // Activation is a column vector
        // Same dimensions as bias
        this->add_matrix_pairs(_activations_d, cols, shouldRandomize);
    }

    this->init();
    this->write_to_device();
}

vnn::vnn(clwrapper::clcontext& con, const std::string &filename) : model(con) {
    _context._queue.finish();

    std::ifstream in(filename, std::ios::binary | std::ios::in);

    uint16_t matrix_entry_size;
    in.read(BYTE_PTR(matrix_entry_size), sizeof(uint16_t));

    assert(matrix_entry_size == sizeof(VNN_FLOAT_TYPE));

    uint16_t number_of_layers;
    in.read(BYTE_PTR(number_of_layers), sizeof(uint16_t));
    _layers = static_cast<size_t>(number_of_layers);

    _neurons_per_layer.reserve(number_of_layers);
    for(uint16_t i = 0; i < number_of_layers; i++) {
        cl_uint output;
        in.read(BYTE_PTR(output), sizeof(cl_uint));

        _neurons_per_layer.emplace_back(output);
    }

    bool shouldRandomize = false;

    this->add_matrix_pairs(_activations_d, _neurons_per_layer[0], shouldRandomize);

    for(uint16_t i = 1; i < number_of_layers; i++) {
        cl_uint rows = _neurons_per_layer[i-1];
        cl_uint cols = _neurons_per_layer[i];
        cl_uint n = rows * cols;

        this->add_matrix_pairs(_weights_d, n, shouldRandomize);
        this->add_matrix_pairs(_biases_d, cols, shouldRandomize);
        this->add_matrix_pairs(_activations_d, cols, shouldRandomize);

        in.read((char*)_weights_d[MAIN_CL_BUFFERS][i-1].host_data(), n * sizeof(VNN_FLOAT_TYPE));
        in.read((char*)_biases_d[MAIN_CL_BUFFERS][i-1].host_data(), cols * sizeof(VNN_FLOAT_TYPE));

    }

    this->init();
    this->write_to_device();
}

void vnn::init() {

    // Set the NDRange of the kernel to the width of the widest matrix
    // This is to ensure that there are enough threads to compute each matrix in parallel
    uint max_column = *std::max_element(_neurons_per_layer.begin() + 1, _neurons_per_layer.end());
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

    this->run(input, output);

    return output;
}


void vnn::train(
    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& input,
    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>& output,
    uint iterations,
    VNN_FLOAT_TYPE learning_rate
) {
    assert(input.size() == output.size());

    size_t n = input.size();
    size_t input_sz = _neurons_per_layer[0];
    size_t output_sz = _neurons_per_layer[_layers-1];

    for(size_t i = 0; i < n; i++) {
        assert(input[i].size() == input_sz);
        assert(output[i].size() == output_sz);
    }


    for(uint epoch = 1; epoch <= iterations; epoch++) {
        this->zero_gradient();

        for(size_t i = 0; i < n; i++) {
            this->zero_gradient_activations();
            this->forward(input[i].get());
            this->backprop(output[i].get());
        }

        this->apply_gradient( static_cast<cl_uint>(n), static_cast<cl_float>(learning_rate) );
        std::cout << epoch << "/" << iterations << "\n";
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
        this->run(input[i], out);

        for(size_t j = 0; j < out.size(); j++) {
            VNN_FLOAT_TYPE tmp = out[j] - expected_output[i][j];
            err += tmp*tmp;
        }
    }

    return err / static_cast<VNN_FLOAT_TYPE>(n) / static_cast<VNN_FLOAT_TYPE>(_neurons_per_layer[_layers-1]);
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

// Applies gradient stored in GRADIENT_CL_BUFFERS to MAIN_CL_BUFFERS with
// given learning rate.
void vnn::apply_gradient(cl_uint n, VNN_FLOAT_TYPE learning_rate) {

    _apply_gradient_kernel.setArg(6, sizeof(cl_uint), &n);

    for(size_t l = 0; l < _layers-1; l++) {
        _apply_gradient_kernel.setArg(0, _weights_d[MAIN_CL_BUFFERS][l].get());
        _apply_gradient_kernel.setArg(1, _weights_d[GRADIENT_CL_BUFFERS][l].get());

        _apply_gradient_kernel.setArg(2, _biases_d[MAIN_CL_BUFFERS][l].get());
        _apply_gradient_kernel.setArg(3, _biases_d[GRADIENT_CL_BUFFERS][l].get());

        cl_uint rows = _neurons_per_layer[l];
        cl_uint cols = _neurons_per_layer[l+1];

        _apply_gradient_kernel.setArg(4, sizeof(cl_uint), &cols);
        _apply_gradient_kernel.setArg(5, sizeof(cl_uint), &rows);
        _apply_gradient_kernel.setArg(7, sizeof(cl_float), &learning_rate);

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

void vnn::add_matrix_pairs(
    std::array<std::vector<clwrapper::memory<VNN_FLOAT_TYPE>>, 2> &out,
    cl_uint n, bool shouldRandomize) {

    out[MAIN_CL_BUFFERS].emplace_back(
        clwrapper::memory<VNN_FLOAT_TYPE>(_context, shouldRandomize, static_cast<size_t>(n))
    );

    out[GRADIENT_CL_BUFFERS].emplace_back(
        clwrapper::memory<VNN_FLOAT_TYPE>(_context, shouldRandomize, static_cast<size_t>(n))
    );

}

void vnn::read_from_device() {
    bool shouldBlock = false;
    std::for_each(ALL(_weights_d[MAIN_CL_BUFFERS]), [shouldBlock](clwrapper::memory<VNN_FLOAT_TYPE> &x) {
        x.read_from_device(shouldBlock);
    });

    std::for_each(ALL(_biases_d[MAIN_CL_BUFFERS]), [shouldBlock](clwrapper::memory<VNN_FLOAT_TYPE> &x) {
        x.read_from_device(shouldBlock);
    });
}

void vnn::write_to_device() {
    bool shouldBlock = false;
    std::for_each(ALL(_weights_d[MAIN_CL_BUFFERS]), [shouldBlock](clwrapper::memory<VNN_FLOAT_TYPE> &x) {
        x.write_to_device(shouldBlock);
    });

    std::for_each(ALL(_biases_d[MAIN_CL_BUFFERS]), [shouldBlock](clwrapper::memory<VNN_FLOAT_TYPE> &x) {
        x.write_to_device(shouldBlock);
    });
}

void vnn::serialize(const std::string &filename) {
    read_from_device();
    _context._queue.finish();

    std::ofstream out(filename, std::ios::binary | std::ios::out);

    uint16_t matrix_entry_size = sizeof(VNN_FLOAT_TYPE);
    uint16_t number_of_layers = static_cast<uint16_t>(_layers);

    out.write(BYTE_PTR(matrix_entry_size), sizeof(uint16_t));
    out.write(BYTE_PTR(number_of_layers), sizeof(uint16_t));

    for(uint i = 0; i < number_of_layers; i++) {
        out.write(BYTE_PTR(_neurons_per_layer[i]), sizeof(cl_uint));
    }

    char *ptr;
    size_t len;
    for(uint16_t i = 0; i < number_of_layers-1; i++) {
        ptr = (byte*)_weights_d[MAIN_CL_BUFFERS][i].host_data();
        len = _weights_d[MAIN_CL_BUFFERS][i].size();

        out.write(ptr, sizeof(VNN_FLOAT_TYPE)*len);

        ptr = (byte*)_biases_d[MAIN_CL_BUFFERS][i].host_data();
        len = _biases_d[MAIN_CL_BUFFERS][i].size();

        out.write(ptr, sizeof(VNN_FLOAT_TYPE)*len);
    }
}

