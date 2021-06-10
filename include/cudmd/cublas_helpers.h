#ifndef CUDMD_CUBLAS_HELPERS_H
#define CUDMD_CUBLAS_HELPERS_H

#include <cublas_v2.h> // cublas*

#include <cudmd/error_handling.h> // basic_error
#include <cudmd/handle.h> // basic_handle

using cublas_error = basic_error<cublasStatus_t>;
using cublas_handle = basic_handle<cublasHandle_t, cublasStatus_t,
    cublasCreate, cublasDestroy>;

#endif
