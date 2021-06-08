#ifndef COMMON_H
#define COMMON_H

typedef signed char schar;
typedef unsigned char uchar;
typedef short shrt;
typedef unsigned short ushrt;
typedef unsigned uint;
typedef unsigned long ulong;
typedef long long llong;
typedef unsigned long long ullong;
typedef float flt;
typedef double dbl;
typedef long double ldbl;

#define EXIT_IF(condition, message) \
    do { \
        const char * const ptr = (message);
        if ((condition)) { \
            if (errno) \
                fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, ptr, \
                    strerror(errno)); \
            else \
                fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, ptr); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_CHECK(error) \
    do { \
        const cudaError_t result = (error); \
        if (result != cudaSuccess) { \
            fprintf(stderr, "%s:%d: cuda: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(result)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(status) \
    do { \
        const cublasStatus_t result = (status); \
        if (result != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "%s:%d: cublas: %d\n", __FILE__, __LINE__, \
                result); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUSOLVER_CHECK(status) \
    do { \
        const cusolverStatus_t result = (status); \
        if (result != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "%s:%d: cusolver: %d\n", __FILE__, __LINE__, \
                result); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif
