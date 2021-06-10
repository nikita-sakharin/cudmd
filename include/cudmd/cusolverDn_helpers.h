#ifndef CUDMD_SOLVER_DN_HELPERS_H
#define CUDMD_SOLVER_DN_HELPERS_H

#include <cusolverDn.h> // cusolverDn*, cusolverStatus_t

#include <cudmd/error_handling.h> // basic_error
#include <cudmd/handle.h> // basic_handle

using cusolver_error = basic_error<cusolverStatus_t>;
using cusolverDn_handle = basic_handle<cusolverDnHandle_t, cusolverStatus_t,
    cusolverDnCreate, cusolverDnDestroy>;
using cusolverDn_params = basic_handle<cusolverDnParams_t, cusolverStatus_t,
    cusolverDnCreateParams, cusolverDnDestroyParams>;

#endif
