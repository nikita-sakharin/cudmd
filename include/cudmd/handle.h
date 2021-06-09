#ifndef CUDMD_HANDLE_H
#define CUDMD_HANDLE_H

#include <exception> // exception
#include <iostream> // cerr, endl
#include <type_traits> // is_scalar

template<typename Handle, typename Create, typename Destroy>
class basic_handle final {
public:
    __host__ inline basic_handle();
    __host__ inline basic_handle(const basic_handle &) noexcept = delete;
    __host__ inline basic_handle(basic_handle &&) noexcept = delete;
    __host__ inline basic_handle &operator=(
        const basic_handle &) noexcept = delete;
    __host__ inline basic_handle &operator=(
        basic_handle &&) noexcept = delete;
    __host__ inline ~basic_handle() noexcept;

    __host__ inline Handle handle() const noexcept;

private:
    static_assert(std::is_scalar<Handle>::value,
        "template argument Handle must be scalar"
    );
    static_assert(
        std::is_invocable<Create, Handle *>,
        "Create must be invocable with args: (Handle *)"
    );
    static_assert(
        std::is_invocable<Destroy, Handle>,
        "Destroy must be invocable with args: (Handle)"
    );

    Handle handle_;
};

template<typename Handle>
inline basic_handle<Handle>::basic_handle() {
    throw_if_error(Create(&handle_), "basic_handle::basic_handle: Create");
}

template<typename Handle>
inline basic_handle<Handle>::~basic_handle() noexcept {
    using std::cerr, std::endl, std::exception;
 
    try {
    	throw_if_error(Destroy(handle_),
    	    "basic_handle::~basic_handle: Destroy"
    	);
    } catch (const exception &except) {
#	ifndef NDEBUG
        cerr << except.what() << endl;
#       endif
    }
}

template<typename Handle>
inline basic_handle<Handle>::handle() const noexcept {
    return handle_;
}

#endif
