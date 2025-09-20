# opencl-nn

WIP hobbyist Neural Network in OpenCL/C++.

It is currently capable of solving the XOR problem. The demo for that can be
seen in `demo/xor.cpp`.

## Building and running the XOR demo.

Dependencies are:
- cmake
- a C++ compiler with C++20 support
- OpenCL library AND C++ header for OpenCL
- OpenCL runtime for your GPU


If you get a bad fit, then just run it again. The initial parameters are
randomized.

```
mkdir build
cd ./build
cmake ..
cmake --build . -t runxor
```

Note that the working directory path matters as the OpenCL kernel code is
compiled at runtime. Running the program via CMake ensures the
correct directory is set(which is the root of the project).

## Future goals

The big end goal is to train a model on the MNIST data set(handwritten digits).

A bit of testing has shown that the current implementation is capable of
bringing the cost down after doing a few iterations, but it is *very* slow.

Therefore, the plan is to implement stochastic descent to hopefully provide
better performance. Along with a serialization feature so that good models can
be saved and so training can be paused/resumed at any time.

Oh, and a cleaner way of setting up training data. Because the current solution
is just hideous.

