
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "lazyml.hpp"

using namespace lazyml;

int main() {
    const time_t s = time(0);
    srand((s % 2 == 0 ? s+1 : s)); 

    cl::Device default_device = utils::value_or_panic(clwrapper::getBestDevice(), "Could not any find device");
    clwrapper::clcontext con = {default_device};

    // Just need to make this less hideous
    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> inputs; inputs.reserve(4);
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0,0}) );
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0,1}) );
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1,0}) );
    inputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1,1}) );
    std::for_each(ALL(inputs), [](auto &x) { x.writeToDevice(false);  });

    std::vector<clwrapper::memory<VNN_FLOAT_TYPE>> outputs; outputs.reserve(4);

    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0}) );
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1}) );
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {1}) );
    outputs.emplace_back( clwrapper::memory<VNN_FLOAT_TYPE>(con, {0}) );
    std::for_each(ALL(outputs), [](auto &x) { x.writeToDevice(false);  });

    std::vector<cl_uint> arch = {2, 2, 1};
    models::vnn nn {con, arch};

    std::cout << "COST: " << nn.cost(inputs, outputs) << std::endl;

//    for(const float x : in.row(0)) {
//        std::cout << x << std::endl;
//    }

    std::cout << "0 0 = " << (nn.run(inputs[0]))[0] << std::endl;
    std::cout << "0 1 = " << (nn.run(inputs[1]))[0] << std::endl;
    std::cout << "1 0 = " << (nn.run(inputs[2]))[0] << std::endl;
    std::cout << "1 1 = " << (nn.run(inputs[3]))[0] << std::endl;

    nn.train(inputs, outputs, 10000);
//
    std::cout << "COST: " << nn.cost(inputs, outputs) << std::endl;
//
    std::cout << "0 0 = " << (nn.run(inputs[0]))[0] << std::endl;
    std::cout << "0 1 = " << (nn.run(inputs[1]))[0] << std::endl;
    std::cout << "1 0 = " << (nn.run(inputs[2]))[0] << std::endl;
    std::cout << "1 1 = " << (nn.run(inputs[3]))[0] << std::endl;


    // Create context 
//    cl::Context context({default_device});
//
//    // Vectors A and B we want to add, each of length SIZE(10)
//    // _h convention specifies that these are buffers on the host
//    #define SIZE 10
//    int A_h[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//    int B_h[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
//    int C_h[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
//    int D_h[SIZE] = {0};
//
//    // Read-only memory buffers in VRAM with enough room for the two vectors
//    // CL_MEM_READ_ONLY allows the GPU to only read from these buffers
//    // _d convention specifies that these buffers are on the device(GPU)
//    cl::Buffer A_d(context, CL_MEM_READ_ONLY, sizeof(int)*SIZE);
//    cl::Buffer B_d(context, CL_MEM_READ_ONLY, sizeof(int)*SIZE);
//    cl::Buffer C_d(context, CL_MEM_READ_ONLY, sizeof(int)*SIZE);
//
//    // Output bufffer with enough room to fit vector of length SIZE
//    // CL_MEM_WRITE_ONLY allows the GPU to only write to this buffer
//    cl::Buffer D_d(context, CL_MEM_WRITE_ONLY, sizeof(int)*SIZE);
//
//    // Queue which takes in commands from host and performs them on device
//    cl::CommandQueue queue( context, default_device );
//
//    queue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(int)*SIZE, A_h);
//    queue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(int)*SIZE, B_h);
//    queue.enqueueWriteBuffer(C_d, CL_TRUE, 0, sizeof(int)*SIZE, C_h);
//
//    // Sources manages kernel code
//    cl::Program::Sources sources;
//    std::string kernel_code = utils::file_to_string("cl/simple_add.cl");
//
//    sources.push_back({kernel_code.c_str(), kernel_code.length()});
//
//    cl::Program program(context, sources);
//    if(program.build({default_device}) != CL_SUCCESS) {
//        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
//        exit(-1);
//    }
//
//    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> simple_add(cl::Kernel(program, "simple_add"));
//    cl::NDRange global(SIZE);
//    simple_add(cl::EnqueueArgs(queue, global), A_d, B_d, C_d, D_d).wait();
//
//    queue.enqueueReadBuffer(D_d, CL_TRUE, 0, sizeof(int)*SIZE, D_h);
//
//    for(size_t i = 0; i < SIZE; i++) {
//        std::cout << D_h[i] << " ";
//    }
//    std::cout << std::endl;

}

