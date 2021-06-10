#ifndef CUDMD_SOLVER_HELPERS_H
#define CUDMD_SOLVER_HELPERS_H

#include <>

using solver_error = basic_error<solverStatus_t>;
using solver_handle = basic_handle<solverHandle_t, solverStatus_t,
    solverCreate, solverDestroy
>;

#endif
