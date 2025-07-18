
#include "model/vnn.hpp"
#include "utils.hpp"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/opencl.hpp>

#include <algorithm>
#include <numeric>

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
    this->add_matrix(_activations_d[0], {1, arch[0]});
    this->add_matrix(_activations_d[1], {1, arch[0]});

    for(size_t i = 1; i < n; i++) {
        assert(arch[i] != 0 && "Neuron layer cannot have 0 neurons");

        // Add new weight matrix and randomize the initial weights
        // Number of rows corresponds to the number of columns in the previous activation column vector
        // Number of columns corresponds to the number of neurons in the current layer
        math::matrix w = {arch[i-1], arch[i]}; w.randomize();
        this->add_matrix(_weights_d[0], w);
        this->add_matrix(_weights_d[1], w);

        // Add new bias and randomize the initial biases
        // Bias is a column vector with size corresponding to number of neurons in the current layer
        math::matrix b = {1, arch[i]}; b.randomize();
        this->add_matrix(_biases_d[0], b);
        this->add_matrix(_biases_d[1], b);

        // Activation is a column vector
        // Same dimensions as bias
        math::matrix a = {1, arch[i]};
        this->add_matrix(_activations_d[0], a);
        this->add_matrix(_activations_d[1], a);
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

}

vnn::~vnn() {}


math::matrix<VNN_FLOAT_TYPE> vnn::run(math::matrix<VNN_FLOAT_TYPE> input) {
    // Forward network
    forward(input);

    // Create output buffer
    math::matrix<VNN_FLOAT_TYPE> out = {1, _neurons_per_layer[_layers-1], true};

    // Read output from last activation layer
    _context._queue.enqueueReadBuffer(_activations_d[0][_layers-1], CL_TRUE, 0, sizeof(VNN_FLOAT_TYPE)*out.cols(), out.data().data());

    return out;
}

void vnn::train(math::matrix<VNN_FLOAT_TYPE> input, math::matrix<VNN_FLOAT_TYPE> output, uint iterations) {
    assert(input.cols() == _neurons_per_layer[0] && output.cols() == _neurons_per_layer[_layers-1]);
    assert(input.rows() == output.rows());

    zero_gradient();

    size_t n = input.rows();
    for(size_t i = 0; i < n; i++) {
        zero_gradient_activations();

        forward(input.row(i));
        backprop(output.row(i));
    }

    apply_gradient( static_cast<cl_uint>(n) );
}

VNN_FLOAT_TYPE vnn::cost(math::matrix<VNN_FLOAT_TYPE> input, math::matrix<VNN_FLOAT_TYPE> output) {
    assert(input.cols() == _neurons_per_layer[0] && output.cols() == _neurons_per_layer[_layers-1]);
    assert(input.rows() == output.rows());

    size_t n = input.rows();
    VNN_FLOAT_TYPE err = 0.0f;

    for(size_t i = 0; i < n; i++) {
        math::matrix<VNN_FLOAT_TYPE> out = run(input.row(i));
        math::matrix<VNN_FLOAT_TYPE> expected = output.row(i);

        out -= expected;
        for(size_t j = 0; j < out.cols(); j++) err += out[{0, j}]*out[{0, j}];
    }

    return err / static_cast<float>(n);
}


void vnn::forward(math::matrix<VNN_FLOAT_TYPE> input) {
    assert(input.rows() == 1 && "Expected 1xM matrix where M is the number of input neurons");
    assert(input.cols() == _neurons_per_layer[0]);

    // Copy input to input layer
    _context._queue.enqueueWriteBuffer(_activations_d[0][0], CL_TRUE, 0, sizeof(VNN_FLOAT_TYPE)*input.cols(), input.data().data());

    for(size_t i = 0; i < _layers-1; i++) {

        // arg[0] = weight matrix
        _forward_kernel.setArg(0, _weights_d[0][i]);
        // arg[1] = bias matrix
        _forward_kernel.setArg(1, _biases_d[0][i]);
        // arg[2] = activation matrix
        _forward_kernel.setArg(2, _activations_d[0][i]);

        // arg[3] = number of rows in weight matrix
        cl_uint rows = static_cast<cl_uint>(_neurons_per_layer[i]);
        _forward_kernel.setArg(3, sizeof(cl_uint), &rows);
        // arg[4] = number of columns in weight matrix
        cl_uint cols = static_cast<cl_uint>(_neurons_per_layer[i+1]);
        _forward_kernel.setArg(4, sizeof(cl_uint), &cols);

        _forward_kernel.setArg(5, _activations_d[0][i+1]);

        _context._queue.enqueueNDRangeKernel(_forward_kernel, cl::NullRange, _kernel_range);
    }

}

void vnn::backprop(math::matrix<VNN_FLOAT_TYPE> output) {
    assert(output.rows() == 1 && output.cols() == _neurons_per_layer[_layers-1]);
    cl_uint n = static_cast<cl_uint>(output.cols());

    // INIT

    // Write output to last gradient activation layer in order to compute (aL - y)
    _context._queue.enqueueWriteBuffer(
        _activations_d[1][_layers-1], CL_TRUE, 0, sizeof(VNN_FLOAT_TYPE)*output.cols(), output.data().data()
    );

    // Compute (aL - y)
    _backprop_init_kernel.setArg(0, _activations_d[0][_layers-1]);
    _backprop_init_kernel.setArg(1, _activations_d[1][_layers-1]);
    _backprop_init_kernel.setArg(2, sizeof(cl_uint), &n);
    _context._queue.enqueueNDRangeKernel(_backprop_init_kernel, cl::NullRange, _kernel_range);

    for(size_t l = _layers-1; l > 0; l--) {
        // weights matrix and weights gradient
        _backprop_step_kernel.setArg(0, _weights_d[0][l-1]);
        _backprop_step_kernel.setArg(1, _weights_d[1][l-1]);

        // Bias column vector
        _backprop_step_kernel.setArg(2, _biases_d[1][l-1]);

        // Activations matrix of current and previous layer
        _backprop_step_kernel.setArg(3, _activations_d[0][l]);
        _backprop_step_kernel.setArg(4, _activations_d[0][l-1]);

        // Buffers used for intermediary values
        _backprop_step_kernel.setArg(5, _activations_d[1][l]);
        _backprop_step_kernel.setArg(6, _activations_d[1][l-1]);

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
        _apply_gradient_kernel.setArg(0, _weights_d[0][l]);
        _apply_gradient_kernel.setArg(1, _weights_d[1][l]);

        _apply_gradient_kernel.setArg(2, _biases_d[0][l]);
        _apply_gradient_kernel.setArg(3, _biases_d[1][l]);

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

        _zero_kernel.setArg(0, _activations_d[1][l]);
        _zero_kernel.setArg(1, sizeof(cl_uint), &cols);
        _context._queue.enqueueNDRangeKernel(_zero_kernel, cl::NullRange, _kernel_range);
    }
}

void vnn::zero_gradient() {
    for(size_t l = 0; l < _layers-1; l++) {
        cl_uint rows = _neurons_per_layer[l];
        cl_uint cols = _neurons_per_layer[l+1];
        cl_uint n = rows * cols;

        _zero_kernel.setArg(0, _weights_d[1][l]);
        _zero_kernel.setArg(1, sizeof(cl_uint), &n);

        _context._queue.enqueueNDRangeKernel(_zero_kernel, cl::NullRange, _kernel_range);

        _zero_kernel.setArg(0, _biases_d[1][l]);
        _zero_kernel.setArg(1, sizeof(cl_uint), &cols);
        _context._queue.enqueueNDRangeKernel(_zero_kernel, cl::NullRange, _kernel_range);
    }
}


void vnn::add_matrix(std::vector<cl::Buffer> &matrix_list, math::matrix<VNN_FLOAT_TYPE> mat) {
    size_t sz = mat.rows()*mat.cols();
    matrix_list.emplace_back(cl::Buffer(_context._context, CL_MEM_READ_WRITE, sizeof(VNN_FLOAT_TYPE)*sz));
    _context._queue.enqueueWriteBuffer(matrix_list[matrix_list.size()-1], CL_FALSE, 0, sizeof(VNN_FLOAT_TYPE)*sz, mat.data().data());
}
