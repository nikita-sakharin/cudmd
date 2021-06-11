#include <cstddef> // size_t

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <library_types.h> // CUDA_C_64F

#include <thrust/complex.h> // complex
#include <thrust/device_ptr.h> // device_ptr
#include <thrust/device_vector.h> // device_vector
#include <thrust/host_vector.h> // host_vector
#include <thrust/tuple.h> // make_tuple, tuple

#include <cudmd/cublas_helpers.h>
#include <cudmd/cudmd.h>
#include <cudmd/cusolverDn_helpers.h>
#include <cudmd/error_handling.h>
#include <cudmd/types.h>

using std::size_t;
using thrust::complex;
using thrust::device_ptr;
using thrust::device_vector;
using thrust::host_vector;
using thrust::make_tuple;
using thrust::tuple;

__host__ tuple<
    device_vector<dbl>,
    device_vector<complex<dbl>>, device_vector<complex<dbl>>
> cudmd(
    const device_ptr<complex<dbl>> a_ptr,
    const size_t m, const size_t n, const size_t k
) {
    cusolverDn_handle handle;
    cusolverDn_params params;
    const int64_t p = 2 * k, niters = 2;

    device_vector<dbl> s_vector(k);
    device_vector<complex<dbl>> u_vector(m * k), v_vector((n - 1) * k);
    size_t device_size, host_size;
    throw_if_error(cusolverDnXgesvdr_bufferSize(
        handle.handle(), params.handle(),
        'S', 'S', m, n - 1, k, p, niters,
        CUDA_C_64F, a_ptr.get(), m,
        CUDA_R_64F, s_vector.data().get(),
        CUDA_C_64F, u_vector.data().get(), m,
        CUDA_C_64F, v_vector.data().get(), n - 1,
        CUDA_C_64F,
        &device_size, &host_size
    ), "cudmd: cusolverDnXgesvdr_bufferSize");

    device_vector<char> device_workspace(device_size);
    host_vector<char> host_workspace(host_size);
    int info;
    throw_if_error(cusolverDnXgesvdr(
        handle.handle(), params.handle(),
        'S', 'S', m, n - 1, k, p, niters,
        CUDA_C_64F, a_ptr.get(), m,
        CUDA_R_64F, s_vector.data().get(),
        CUDA_C_64F, u_vector.data().get(), m,
        CUDA_C_64F, v_vector.data().get(), n - 1,
        CUDA_C_64F,
        device_workspace.data().get(), device_size,
        host_workspace.data(), host_size,
        &info
    ), "cudmd: cusolverDnXgesvdr");

    return make_tuple(s_vector, u_vector, v_vector);
}
