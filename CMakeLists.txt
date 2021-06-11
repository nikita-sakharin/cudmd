cmake_minimum_required(VERSION 3.16.0)

project(cudmd VERSION 1.0.0 LANGUAGES CUDA CXX)

set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDAFLAGS -Werror
    cross-execution-space-call,deprecated-declarations,reorder
)
add_compile_options("$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:${CUDAFLAGS}>")

add_subdirectory(lib/googletest)
add_subdirectory(src)
add_subdirectory(test)

enable_testing()
add_test(NAME test COMMAND ${PROJECT_NAME}_test)
