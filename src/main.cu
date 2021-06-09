#include <cassert>
#include <cstddef>

#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cudmd/error_handling.h>

using cublas_error = basic_error<cublasStatus_t>;

__device__ inline int sum3(const int a, const int b, const int c) {
    return a + b + c;
}

__global__ __forceinline__ void hcd() {
    const int a = 5, b = 7, c = 11;
    const int d = sum3(a, b, c);
}

int main() {
    cublas_error
    	g(CUBLAS_STATUS_SUCCESS, "4545"),
    	h(CUBLAS_STATUS_SUCCESS, std::string("test")),
    	i(CUBLAS_STATUS_SUCCESS),
    	j(CUBLAS_STATUS_SUCCESS);
    std::cout << g.what() << '\n' << i.code();
    hcd<<<1, 1>>>();

    return 0;
}
