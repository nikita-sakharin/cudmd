#include <cassert>
#include <cstddef>

#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cudmd/error_handling.h>
#include <cudmd/handle.h>

using cublas_error = basic_error<cublasStatus_t>;
using cublas_handle = basic_handle<cublasHandle_t, cublasStatus_t,
    cublasCreate, cublasDestroy
>;

int main() {
    cublas_error
    	g(CUBLAS_STATUS_NOT_INITIALIZED, "4545"),
    	h(CUBLAS_STATUS_INVALID_VALUE, std::string("test")),
    	i(CUBLAS_STATUS_MAPPING_ERROR),
    	j;
    std::cout
    	<< "code: " << g.code() << "\nwhat:\n" << g.what() << '\n'
    	<< "code: " << h.code() << "\nwhat:\n" << h.what() << '\n'
    	<< "code: " << i.code() << "\nwhat:\n" << i.what() << '\n'
    	<< "code: " << j.code() << "\nwhat:\n" << j.what() << '\n';
    std::string what("test");
    std::cout << &what << '\n';
    // throw_if_error(CUBLAS_STATUS_NOT_INITIALIZED, what);
    
    cublas_handle{};

    return 0;
}
