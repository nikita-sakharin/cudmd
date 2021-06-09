#include <cassert>
#include <cstddef>

#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cudmd/error_handling.h>

using cublas_error = basic_error<cublasStatus_t>;

int main() {
    cublas_error
    	g(CUBLAS_STATUS_SUCCESS, "4545"),
    	h(CUBLAS_STATUS_SUCCESS, std::string("test")),
    	i(CUBLAS_STATUS_SUCCESS),
    	j(CUBLAS_STATUS_SUCCESS);
    std::cout << g.what() << '\n' << i.code();

    return 0;
}
