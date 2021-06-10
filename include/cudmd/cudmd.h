#ifndef CUDMD_CUDMD_H
#define CUDMD_CUDMD_H

#include <cstdint> // int64_t

#include <cuComplex.h> // cuDoubleComplex

#include <thrust/device_ptr.h> // device_ptr
#include <thrust/device_vector.h> // device_vector
#include <thrust/tuple.h> // tuple

__host__ thrust::tuple<
    thrust::device_vector<cuDoubleComplex>,
    thrust::device_vector<cuDoubleComplex>,
    thrust::device_vector<cuDoubleComplex>
> cudmd(
    thrust::device_ptr<cuDoubleComplex>,
    std::int64_t, std::int64_t, std::int64_t k,
    std::int64_t = 2 * k, std::int64_t = 2,
);

#endif
