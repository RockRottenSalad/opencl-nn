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
target_compile_options(lazyml PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -DCL_HPP_TARGET_OPENCL_VERSION=300 -g)

add_executable(xor ${DEMO_SOURCE_DIR}/xor.cpp)
target_include_directories(xor PUBLIC ${INCLUDE_DIR})
set_property(TARGET xor PROPERTY DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}") 
target_link_libraries(xor PUBLIC lazyml)
target_compile_options(xor PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -DCL_HPP_TARGET_OPENCL_VERSION=300 -g)

add_executable(mnist ${DEMO_SOURCE_DIR}/mnist.cpp)
target_include_directories(mnist PUBLIC ${INCLUDE_DIR})
set_property(TARGET mnist PROPERTY DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}") 
target_link_libraries(mnist PUBLIC lazyml)
target_compile_options(mnist PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -DCL_HPP_TARGET_OPENCL_VERSION=300 -g)


add_custom_target(runxor COMMAND xor WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_custom_target(runmnist COMMAND mnist WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

