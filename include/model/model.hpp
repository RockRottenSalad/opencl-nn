#pragma once

#include "clwrapper.hpp"

namespace lazyml {

namespace models {

    template<typename T = float>
    class model {
        protected:
        clwrapper::clcontext& _context;
        public:
        model(clwrapper::clcontext& con) :_context(con) {}
        virtual void run(clwrapper::memory<T>& input, std::vector<T> &output) = 0;
        virtual std::vector<T> run(clwrapper::memory<T>& input) = 0;

        virtual void train(
                std::vector<clwrapper::memory<T>>& input,
                std::vector<clwrapper::memory<T>>& output,
                uint iterations,
                T learning_rate
            ) = 0;

        virtual T cost(
                std::vector<clwrapper::memory<T>>& input,
                std::vector<clwrapper::memory<T>>& output
            ) = 0;

        virtual void serialize(
                const std::string &filename
        ) = 0;
    };

}

}
