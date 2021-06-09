#ifndef CUDMD_ERROR_HANDLING_H
#define CUDMD_ERROR_HANDLING_H

#include <stdexcept> // runtime_error
#include <string> // string
#include <type_traits> // is_same_v

#include <cublas_v2.h> //
#include <cusolverDn.h> //

template<class Status>
class basic_error final : public std::runtime_error {
public:
    basic_error(Status, const std::string &);
    basic_error(Status, const char *);
    basic_error(Status);
    basic_error(const basic_error &) noexcept = default;
    basic_error &operator=(const basic_error &) noexcept = default;
    virtual ~basic_error() noexcept;

    Status code() const noexcept;
    const char *what() const noexcept override;

private:
    static_assert(
        std::is_same_v<Status, cublasStatus_t> ||
        std::is_same_v<Status, cusolverStatus_t>,
        "template argument Status must have type either
        "cublasStatus_t or cusolverStatus_t"
    );

    Status status_;
};

template<class E>
inline void throw_if_error(const Status status) {
    
}

#endif
