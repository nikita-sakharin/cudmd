#include <cstdint>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <thrust/device_vector.h>
#include <thrust/tuple.h>

#include <cudmd/cublas_helpers.h>
#include <cudmd/cudmd.h>
#include <cudmd/cusolverDn_helpers.h>

__host__ thrust::tuple<> cudmd(thrust::device_ptr<cuDoubleComplex *> a_ptr,
    const int64_t m, const int64_t n, const int64_t k,
    const int64_t p = 2 * k, const int64_t niters = 2,
) {
    using namespace thrust;

    cusolverDn_handle handle;
    cusolverDn_params params;

    size_t device_size, host_size;
    throw_if_error(cusolverDnXgesvdr_bufferSize(
        handle.handle(), params.handle(),
        'S', 'S',
        m, n, k,
        p, niters,
        CUDA_C_64F, const void *A, int64_t lda,
        CUDA_C_64F, const void *Srand,
        CUDA_C_64F, const void *Urand, int64_t ldUrand,
        CUDA_C_64F, const void *Vrand, int64_t ldVrand,
        CUDA_C_64F,
        &device_size &host_size
    ), "cudmd: cusolverDnXgesvdr_bufferSize");
    thrust::device_vector<> device_workspace();
    thrust::device_vector<> device_workspace();

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
