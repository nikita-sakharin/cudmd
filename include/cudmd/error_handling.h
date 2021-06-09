#ifndef CUDMD_ERROR_HANDLING_H
#define CUDMD_ERROR_HANDLING_H

#include <stdexcept> // runtime_error
#include <string> // string
#include <type_traits> // is_enum

template<typename Code>
class basic_error final : public std::runtime_error {
public:
    __host__ inline basic_error(Code, const std::string &);
    __host__ inline basic_error(Code, const char *);
    __host__ inline basic_error(Code = Code());
    __host__ inline basic_error(const basic_error &) noexcept = default;
    __host__ inline basic_error &operator=(
        const basic_error &) noexcept = default;
    __host__ inline ~basic_error() noexcept override = default;

    __host__ inline Code code() const noexcept;
    __host__ inline const char *what() const noexcept override;

private:
    static_assert(std::is_enum<Code>::value,
        "template argument Code must be enum"
    );

    Code code_;
};

template<typename Code>
__host__ inline basic_error<Code>::basic_error(
    const Code code,
    const std::string &what
) : runtime_error((what.size() > 0 ? what + ": " : what) + std::to_string(code)),
    code_(code) {}

template<typename Code>
__host__ inline basic_error<Code>::basic_error(
    const Code code,
    const char * const what
) : basic_error(code, std::string(what)) {}

template<typename Code>
__host__ inline basic_error<Code>::basic_error(
    const Code code
) : basic_error(code, "") {}

template<typename Code>
__host__ inline Code basic_error<Code>::code() const noexcept {
    return code_;
}

template<typename Code>
__host__ inline const char *basic_error<Code>::what() const noexcept {
    return std::runtime_error::what();
}

template<typename Code, typename... Types>
__host__ inline void throw_if_error(const Code code, const Types &...args) {
    if (code != Code())
        throw basic_error<Code>(code, args...);
}

#endif
