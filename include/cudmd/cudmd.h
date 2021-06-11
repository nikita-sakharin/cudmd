#ifndef CUDMD_CUDMD_H
#define CUDMD_CUDMD_H

#include <cstddef> // size_t

#include <thrust/complex.h> // complex
#include <thrust/device_ptr.h> // device_ptr
#include <thrust/device_vector.h> // device_vector
#include <thrust/tuple.h> // tuple

#include <cudmd/types.h>

__host__ thrust::tuple<
    thrust::device_vector<dbl>,
    thrust::device_vector<thrust::complex<dbl>>,
    thrust::device_vector<thrust::complex<dbl>>
> cudmd(
    thrust::device_ptr<thrust::complex<dbl>>,
    std::size_t, std::size_t, std::size_t
);

#endif
