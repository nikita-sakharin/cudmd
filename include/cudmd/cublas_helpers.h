#ifndef CUDMD_CUBLAS_HELPERS_H
#define CUDMD_CUBLAS_HELPERS_H

#include <>

using cublas_error = basic_error<cublasStatus_t>;
using cublas_handle = basic_handle<cublasHandle_t, cublasStatus_t,
    cublasCreate, cublasDestroy
>;

#endif
