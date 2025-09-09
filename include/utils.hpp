#pragma once

#include<string>
#include<optional>
#include<iostream>
#include<cassert>

#define ALL(container) (container).begin(), (container).end()

template <typename T>
concept IsContainer = 
    requires(T t) {
        t.begin();
        t.end();
        t.data();
        t[0];
        t.size();
    };

namespace lazyml {

namespace utils {

    // Given equation x = 2^y
    // Returns ceil(y)
    //
    // Used to find ideal work group size for kernels
    uint nearest_power_of_two(uint x);

    std::string file_to_string(std::string filepath);

    template<typename T>
    T value_or_panic(const std::optional<T>& opt, const std::string& msg) {
        if(opt) {
            return opt.value();
        }

        std::cout << msg << std::endl;
        exit(-1);
    }

    template<typename T>
    class view {
        private:
            const T *_data;
            const size_t _size, _stride;
        public:
            view(T *data, size_t size, size_t stride) : _data(data), _size(size), _stride(stride) {}
            view(T *data, size_t size) : _data(data), _size(size), _stride(1) {}

            T operator[](size_t index) { assert(index < _size); return _data[index*_stride]; }
            const T* begin() const { return _data; }
            const T* end() const { return _data + _size; }
            const T* data() const { return _data; }
            size_t size() const { return _size; }
    };

}
}

