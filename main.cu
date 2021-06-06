#include <cstddef>

#include <iostream>

#include <cublas_v2.h>
#include <cusolverDn.h>

int main() {
    cusolverDnHandle_t cusolverH;
    // cublasHandle_t cublasH;
    
    cusolverDnCreate(&cusolverH);

    size_t size;
    scanf("%zu", &size);
    int
        *a_host = (int *) malloc(sizeof(int) * size),
        *b_host = (int *) malloc(sizeof(int) * size),
        *c_host = (int *) malloc(sizeof(int) * size);

    for (size_t i = 0; i < size; ++i)
    	scanf("%d", a_host + i);
    for (size_t i = 0; i < size; ++i)
    	scanf("%d", b_host + i);
    
    int *a_device, *b_device, *c_device;
    cudaMalloc(&a_device, sizeof(int) * size);
    cudaMalloc(&b_device, sizeof(int) * size);
    cudaMalloc(&c_device, sizeof(int) * size);
    cudaMemcpy(a_device, a_host, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, sizeof(int) * size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(c_host, c_device, sizeof(int) * size, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < size; ++i)
    	printf("%d ", c_host[i]);
    
    return 0;
}
