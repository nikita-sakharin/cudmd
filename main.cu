#include <stdio.h>

static __global__ void test(
    const int * __restrict__,
    const int * __restrict__,
    int * __restrict__, size_t
);

static __device__ void plus(
    const int * __restrict__,
    const int * __restrict__,
    int * __restrict__, size_t
);

int main(void) {
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

    test<<<32, 32>>>(a_device, b_device, c_device, size);
    
    cudaMemcpy(c_host, c_device, sizeof(int) * size, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < size; ++i)
    	printf("%d ", c_host[i]);
    
    return 0;
}

static __global__ void test(
    const int * const __restrict__ a,
    const int * const __restrict__ b,
    int * const __restrict__ c,
    const size_t size
) {
    const size_t idx = threadIdx.x + blockIdx.x * gridDim.x,
        offset = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += offset)
        plus(a, b, c, i);
}

static inline __device__ void plus(
    const int * const __restrict__ a,
    const int * const __restrict__ b,
    int * const __restrict__ c,
    const size_t index
) {
    c[index] = a[index] + b[index];
}
