#include <cstddef> // size_t

#include <math_constants.h> // CUDART_PI

#include <thrust/complex.h> // complex
#include <thrust/device_vector.h> // device_vector
#include <thrust/generate.h> // generate
#include <thrust/host_vector.h> // host_vector

#include <gtest/gtest.h>

#include <cudmd/cudmd.h>
#include <cudmd/types.h>

using std::size_t;
using thrust::complex;
using thrust::device_vector;
using thrust::generate;
using thrust::host_vector;
using thrust::tie;

TEST(CudmdTest, First) {
    static const size_t x_size = (1U << 10) + 1U, t_size = (1U << 8) + 1U;
    static const complex<dbl> x_start = -5.0, x_stop = 5.0;
    static const dbl t_start = 0.0, t_stop = 4.0 * CUDART_PI;

    host_vector<complex<dbl>> host_x_upper(t_size * x_size);
    generate(host_x_upper.begin(), host_x_upper.end(),
        [n = 0U]() mutable noexcept -> complex<dbl> {
            const size_t i = n / x_size, j = n % x_size;
            const complex<dbl>
                x = x_start + j * (x_stop - x_start) / (x_size - 1U);
            const dbl t = t_start + i * (t_stop - t_start) / (t_size - 1U);
            const complex<dbl>
                f1 = 1.0 / cosh(x + 3.0) * exp(complex<dbl>(0.0, 2.3 * t)),
                f2 = 2.0 / cosh(x) * tanh(x) * exp(complex<dbl>(0.0, 2.8 * t));
            return f1 + f2;
        }
    );
    device_vector<complex<dbl>> device_x_upper = host_x_upper;
    host_vector<dbl> s;
    host_vector<complex<dbl>> u, v;
    tie(s, u, v) = cudmd(device_x_upper.data(), x_size, t_size, 2);
}
