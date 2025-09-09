cmake_minimum_required(VERSION 3.12)

project(lazyml)

set(DEMO_SOURCE_DIR "./demo/")
set(LAZYML_SOURCE_DIR "./src/")
set(INCLUDE_DIR "./include/")

set(CMAKE_CXX_STANDARD "20")
set(CMAKE_EXE_LINKER_FLAGS  "-lOpenCL -lm")

# ADD LAZYML SOURCE FILES HERE
set(LAZYML_FILES "clwrapper.cpp" "kernels.cpp" "utils.cpp" "model/vnn.cpp")


list(TRANSFORM LAZYML_FILES PREPEND ${LAZYML_SOURCE_DIR})
add_library(lazyml STATIC ${LAZYML_FILES})
target_include_directories(lazyml PUBLIC ${INCLUDE_DIR})
target_compile_options(lazyml PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -DCL_HPP_TARGET_OPENCL_VERSION=300 -O3)

add_executable(demo ${DEMO_SOURCE_DIR}/main.cpp)
target_include_directories(demo PUBLIC ${INCLUDE_DIR})
set_property(TARGET demo PROPERTY DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}") 

target_link_libraries(demo PUBLIC lazyml)
target_compile_options(demo PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -DCL_HPP_TARGET_OPENCL_VERSION=300 -O3)


add_custom_target(run COMMAND demo WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

