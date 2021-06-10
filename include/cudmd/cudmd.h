#ifndef CUDMD_CUDMD_H
#define CUDMD_CUDMD_H

#include <thrust/device_ptr.h> // device_ptr
#include <thrust/device_vector.h> // device_vector
#include <thrust/tuple.h> // tuple

__host__ thrust::tuple<
    thrust::device_vector<cuDoubleComplex>,
    thrust::device_vector<cuDoubleComplex>,
    thrust::device_vector<cuDoubleComplex>
> cudmd(
    device_ptr<cuDoubleComplex>,
    int64_t, int64_t, int64_t k,
    int64_t = 2 * k, int64_t = 2,
);

#endif
