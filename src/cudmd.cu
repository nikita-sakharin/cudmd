#include <cstddef> // size_t
#include <cstdint> // int64_t

#include <cublas_v2.h>
#include <cuComplex.h> // cuDoubleComplex
#include <cusolverDn.h>
#include <library_types.h> // CUDA_C_64F

#include <thrust/device_ptr.h> // device_ptr
#include <thrust/device_vector.h> // device_vector
#include <thrust/host_vector.h> // host_vector
#include <thrust/tuple.h> // tuple

#include <cudmd/error_handling.h> // throw_if_error
#include <cudmd/cublas_helpers.h>
#include <cudmd/cudmd.h> // cudmd
#include <cudmd/cusolverDn_helpers.h> // cusolverDn_handle, cusolverDn_params

using std::int64_t;
using std::size_t;
using thrust::device_ptr;
using thrust::tuple;

__host__ tuple<> cudmd(device_ptr<cuDoubleComplex> a_ptr,
    const int64_t m, const int64_t n, const int64_t k,
    const int64_t p = 2 * k, const int64_t niters = 2,
) {
    cusolverDn_handle handle;
    cusolverDn_params params;

    device_vector<cuDoubleComplex> s_vector(k),
        u_vector(m * k), v_vector(n * k);
    size_t device_size, host_size;
    throw_if_error(
        cusolverDnXgesvdr_bufferSize(
            handle.handle(), params.handle(),
            'S', 'S', m, n, k, p, niters,
            CUDA_C_64F, a_ptr.get(), int64_t lda,
            CUDA_C_64F, s_vector.get(),
            CUDA_C_64F, u_vector.get(), int64_t ldUrand,
            CUDA_C_64F, u_vector.get(), int64_t ldVrand,
            CUDA_C_64F,
            &device_size, &host_size
        ),
        "cudmd: cusolverDnXgesvdr_bufferSize"
    );
    device_vector<char> device_workspace(device_size);
    host_vector<char> host_workspace(host_size);

    throw_if_error(cusolverDnXgesvdr(
        handle.handle(),
        params.handle(),
        signed char jobu,
        signed char jobv,
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t p,
        int64_t niters,
        CUDA_C_64F,
        void *A,
        int64_t lda,
        CUDA_C_64F,
        void *Srand,
        CUDA_C_64F,
        void *Urand,
        int64_t ldUrand,
        CUDA_C_64F,
        void *Vrand,
        int64_t ldVrand,
        CUDA_C_64F,
        void *bufferOnDevice,
        size_t workspaceInBytesOnDevice,
        void *bufferOnHost,
        size_t workspaceInBytesOnHost,
        int *d_info
    ), "cudmd: cusolverDnXgesvdr");
}
