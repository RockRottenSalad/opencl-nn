#pragma once

#include "kernels.hpp"
#include "math/math.hpp"
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
    
    // TODO clean this up
    template<typename T>
    class memory {
        public:
            memory(clcontext &context, std::initializer_list<T> initial_values) : 
            _context(context),
            _host(initial_values),
            _device(cl::Buffer(_context._context, CL_MEM_READ_WRITE, sizeof(T)*initial_values.size()))
            {}

            memory(clcontext &context, std::span<T> initial_values) : 
            _context(context),
            _device(cl::Buffer(_context._context, CL_MEM_READ_WRITE, sizeof(T)*initial_values.size()))
            {
                size_t n = initial_values.size();
                _host = std::vector<T>(n, 0);
                for(size_t i = 0; i < n; i++) _host[i] = initial_values[i];
            }

            memory(clcontext &context, bool random, size_t n) : 
            _context(context)
            {
                _host = std::vector<T>(n, 0);

                if(random) for(size_t i = 0; i < n; i++) _host[i] = math::rand_float();
                else for(size_t i = 0; i < n; i++) _host[i] = 0;

                _device = cl::Buffer(_context._context, CL_MEM_READ_WRITE, sizeof(T)*_host.size());
            }

            void writeToDevice(bool blocking) { 
                size_t zero_offset = 0;
                _context._queue.enqueueWriteBuffer(
                    _device,
                    (blocking ? CL_TRUE : CL_FALSE), // <-- Need to convert std bool to cl_bool
                    zero_offset,
                    sizeof(T)*_host.size(),
                    _host.data()
                ); 
            }

            void readFromDevice(bool blocking) { 
                size_t zero_offset = 0;
                _context._queue.enqueueReadBuffer(
                    _device,
                    (blocking ? CL_TRUE : CL_FALSE),
                    zero_offset,
                    sizeof(T)*_host.size(),
                    _host.data()
                ); 
            }

            cl::Buffer& get() { return _device; }

            T& operator[](size_t index) { return _host[index]; }
            T* host_data() { return _host.data(); }

            size_t size() { return _host.size(); }
        private:
            clcontext& _context;
            std::vector<T> _host;
            cl::Buffer _device;
    };
}

}
