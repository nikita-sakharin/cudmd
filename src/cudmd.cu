#include <cstddef> // size_t

#include <stdexcept> // out_of_range

#include <cublas_v2.h>
#include <cusolverDn.h> // cusolverDnXgesvd*
#include <library_types.h> // CUDA_C_64F, CUDA_R_64F

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

using std::out_of_range;
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
    const size_t m, const size_t n, const size_t rank
) {
    const size_t k = n - 1;
    if (rank > k) [[unlikely]]
        throw out_of_range("cudmd: n must be greater than rank");

    cusolverDn_handle handle;

    device_vector<dbl> s_vector(k);
    device_vector<complex<dbl>> u_vector(m * k), vt_vector(k * k);
    size_t device_size, host_size;
    throw_if_error(cusolverDnXgesvd_bufferSize(
        handle.handle(), nullptr,
        'S', 'S', m, k,
        CUDA_C_64F, a_ptr.get(), m,
        CUDA_R_64F, s_vector.data().get(),
        CUDA_C_64F, u_vector.data().get(), m,
        CUDA_C_64F, vt_vector.data().get(), k,
        CUDA_C_64F,
        &device_size, &host_size
    ), "cudmd: cusolverDnXgesvd_bufferSize");

    device_vector<char> device_workspace(device_size);
    host_vector<char> host_workspace(host_size);
    device_vector<int> device_info(1);
    throw_if_error(cusolverDnXgesvd(
        handle.handle(), nullptr,
        'S', 'S', m, k,
        CUDA_C_64F, a_ptr.get(), m,
        CUDA_R_64F, s_vector.data().get(),
        CUDA_C_64F, u_vector.data().get(), m,
        CUDA_C_64F, vt_vector.data().get(), k,
        CUDA_C_64F,
        device_workspace.data().get(), device_size,
        host_workspace.data(), host_size,
        device_info.data().get()
    ), "cudmd: cusolverDnXgesvd");
    
    s_vector.resize(rank);
    u_vector.resize(m * rank);

    return make_tuple(s_vector, u_vector, vt_vector);
}
