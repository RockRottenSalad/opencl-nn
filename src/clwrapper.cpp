
#include "clwrapper.hpp"
#include <algorithm>

using namespace lazyml;
using namespace lazyml::clwrapper;

bool less(cl::Device a, cl::Device b, SearchBy searchBy) {
    cl_int x;
    switch(searchBy) {
            case SearchBy::VRAM: 
            return a.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &x) <= b.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &x);
            case SearchBy::FREQ: 
            return a.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &x) <= b.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &x);
    }

    // All cases covered by switch, we will never hit this case.
    // Doesn't mean the compiler won't complain about it though.
    return false;
}

std::optional<cl::Device> getMaxByMemory(std::vector<cl::Device> &devices) {
    auto max_it = std::max_element(devices.begin(), devices.end(), [&](cl::Device &a, cl::Device &b) {
        return less(a, b, SearchBy::VRAM);
    });
    return *max_it;
}

std::optional<cl::Device> getMaxByFrequency(std::vector<cl::Device> &devices) {
    auto max_it = std::max_element(devices.begin(), devices.end(), [&](cl::Device &a, cl::Device &b) {
        return less(a, b, SearchBy::FREQ);
    });

    return *max_it;
}

std::optional<cl::Device> clwrapper::getBestDevice(SearchBy searchBy) {

    // Fetch all OpenCL platforms
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);


    // Holds all devices for current platform
    std::vector<cl::Device> all_devices;

    std::optional<cl::Device> candidate = std::nullopt;
    for(const cl::Platform& platform : all_platforms) {
        platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        if(all_devices.size() == 0) continue;

        std::optional<cl::Device> tmp;
        switch(searchBy) {
            case SearchBy::VRAM: 
                tmp = getMaxByMemory(all_devices);
                break;
            case SearchBy::FREQ: 
                tmp = getMaxByFrequency(all_devices);
                break;
        }

        if(!candidate.has_value()) candidate = tmp;
        else if(tmp.has_value() && less(tmp.value(), candidate.value(), searchBy)) candidate = tmp;
    }

    all_platforms.clear();
    all_devices.clear();

    return candidate;
}


