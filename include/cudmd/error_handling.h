#ifndef CUDMD_ERROR_HANDLING_H
#define CUDMD_ERROR_HANDLING_H

#include <stdexcept> // runtime_error
#include <string> // string

template<class Status>
class cuda_error final : public std::runtime_error {
public:
    cuda_error(Status, const std::string &);
    cuda_error(Status, const char *);
    cuda_error(Status);
    cuda_error(const cuda_error &) noexcept = default;
    cuda_error &operator=(const cuda_error &) noexcept = default;
    virtual ~cuda_error() noexcept;

    Status code() const noexcept;
    const char *what() const noexcept override;

private:
    Status error;
};

template<class E>
void throw_if_error(const E& e) {
}

#endif
