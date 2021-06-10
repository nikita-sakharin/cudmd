#ifndef CUDMD_HANDLE_H
#define CUDMD_HANDLE_H

#include <exception> // exception
#include <iostream> // cerr, endl
#include <type_traits> // is_scalar

#include <cudmd/error_handling.h>

template<typename Handle, typename Status,
    Status (*Create)(Handle *), Status (*Destroy)(Handle)>
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

    Handle handle_;
};

template<typename Handle, typename Status,
    Status (*Create)(Handle *), Status (*Destroy)(Handle)>
__host__ inline basic_handle<Handle, Status, Create, Destroy>::basic_handle() {
    throw_if_error(Create(&handle_), "basic_handle::basic_handle: Create");
}

template<typename Handle, typename Status,
    Status (*Create)(Handle *), Status (*Destroy)(Handle)>
__host__ inline basic_handle<Handle, Status, Create, Destroy>::~basic_handle(
) noexcept {
    using std::cerr; using std::endl; using std::exception;

    try {
    	throw_if_error(Destroy(handle_),
    	    "basic_handle::~basic_handle: Destroy"
    	);
    } catch (const exception &except) {
#	ifndef NDEBUG
        cerr << except.what() << endl;
#       endif
    } catch (...) {}
}

template<typename Handle, typename Status,
    Status (*Create)(Handle *), Status (*Destroy)(Handle)>
__host__ inline Handle basic_handle<Handle, Status, Create, Destroy>::handle(
) const noexcept {
    return handle_;
}

#endif
