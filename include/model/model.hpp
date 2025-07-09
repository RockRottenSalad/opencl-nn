#pragma once

#include "clwrapper.hpp"
#include "math/math.hpp"

namespace lazyml {

namespace models {

    template<typename T = float>
    class model {
        protected:
        clwrapper::clcontext& _context;
        public:
        model(clwrapper::clcontext& con) :_context(con) {}
        virtual math::matrix<T> run(math::matrix<T> input) = 0;
        virtual void train(math::matrix<T> input, math::matrix<T> output, uint iterations) = 0;
        virtual float cost(math::matrix<T> input, math::matrix<T> output) = 0;
    };

}

}
