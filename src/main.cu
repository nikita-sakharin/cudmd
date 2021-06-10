#include <cassert>
#include <cstddef>

#include <iostream>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cudmd/cublas_helpers.h>
#include <cudmd/cusolverDn_helpers.h>

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
    
    cublas_handle l;

    return 0;
}
