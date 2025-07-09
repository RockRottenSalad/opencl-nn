#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include<stddef.h>
#include<vector>
#include<ostream>

#include "utils.hpp"

namespace lazyml {

namespace math {


    // Random floating point number between 0 and 1
    static float rand_float() {
        return (float)rand() / (float)RAND_MAX;
    }

    template<typename T = float>
    class matrix {

    class vector_view {

    };

        private:
        size_t _rows, _cols;
        std::vector<T> _data;
        public:

        matrix(size_t rows, size_t cols, bool zeroed = false) : _rows(rows), _cols(cols) {
            _data.resize(_rows*_cols);
            if(zeroed) {
                size_t n = _rows * _cols;
                for(size_t i = 0; i < n; i++) _data[i] = 0;
//                std::fill(ALL(_data), 0);
            }
        }
        matrix(const matrix &other) : _rows(other._rows), _cols(other._cols) {
            _data.resize(_rows*_cols);

                size_t n = _rows * _cols;
                for(size_t i = 0; i < n; i++) _data[i] = other._data[i];
//            std::copy(ALL(other._data), std::back_inserter(_data));
        }

//        matrix() : _rows(0), _cols(0) {}

        size_t rows() { return _rows; }
        size_t cols() { return _cols; }

        void randomize() {
            const size_t n = _rows*_cols;
            for(size_t i = 0; i < n; i++) {
                _data[i] = (T)std::rand() / (T)RAND_MAX;
            }
        }

        std::vector<T>& data() {
            return _data;
        }

        // frend :)
        friend std::ostream &operator<<(std::ostream &str, matrix &m) { 
            for(size_t r = 0; r < m._rows; r++) {
                for(size_t c = 0; c < m._cols; c++) {
                    str << m[{r,c}] << " ";
                }
                str << "\n";
            }
            return str;
        }

        T& operator[](const std::pair<size_t, size_t> &index) {
            assert(index.first < _rows && "Index out of bounds");
            assert(index.second < _cols && "Index out of bounds");

            return _data[index.first * _cols + index.second];
        }

        void operator*=(matrix &other) {
            assert(_cols == other._rows && "Matrix multiplication inner dimensions don't match");

            for(size_t r = 0; r < _rows; r++) {
                for(size_t c = 0; c < _cols; c++) {
                    T result = 0;
                    for(size_t j = 0; j < _cols; j++) {
                        result += this[{r,j}] * this[{j,c}];
                    }
                    this[{r,c}] = result;
                }
            }
        }

        void operator*=(T scalar) {
            const size_t n = _rows*_cols;
            for(size_t i = 0; i < n; i++) {
                _data[i] *= scalar;
            }
        }

        void operator+=(matrix &other) {
            assert(_rows == other._rows && "Matrix dimensions don't match");
            assert(_cols == other._cols && "Matrix dimensions don't match");

            const size_t n = _rows*_cols;
            for(size_t i = 0; i < n; i++) {
                _data[i] += other._data[i];
            }
        }
        void operator-=(matrix &other) {
            assert(_rows == other._rows && "Matrix dimensions don't match");
            assert(_cols == other._cols && "Matrix dimensions don't match");

            const size_t n = _rows*_cols;
            for(size_t i = 0; i < n; i++) {
                _data[i] -= other._data[i];
            }

        }

        void operator+=(T scalar) {
            const size_t n = _rows*_cols;
            for(size_t i = 0; i < n; i++) {
                _data[i] += scalar;
            }
        }
        void operator-=(T scalar) {
            const size_t n = _rows*_cols;
            for(size_t i = 0; i < n; i++) {
                _data[i] -= scalar;
            }
        }

        matrix<T> row(size_t r) {
            assert(r < _rows && "Requsted row number greater than matrix dimensions");
            matrix<T> ret = {1, _cols};
            for(size_t c = 0; c < _cols; c++)
                ret[{0,c}] = (*this)[{r,c}];

            return ret;
        }

        matrix<T> col(size_t c) {
            assert(c < _cols && "Requsted column number greater than matrix dimensions");
            matrix<T> ret = {_rows, 1};
            for(size_t r = 0; r < _rows; r++)
                ret[{r,0}] = (*this)[{r,c}];
            return ret;
        }

    };

}

}
