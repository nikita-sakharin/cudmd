set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

set(TARGET ${PROJECT_NAME})

add_executable(${TARGET} main.cu
    cudmd.cu
)
