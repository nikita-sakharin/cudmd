set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

set(TARGET ${PROJECT_NAME})

add_executable(${TARGET} main.cu
    cudmd.cu
)
target_include_directories(${TARGET} PRIVATE "${PROJECT_SOURCE_DIR}/include")
