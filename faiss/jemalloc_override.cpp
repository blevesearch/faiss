/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifdef USE_JEMALLOC
// ----------------------------------------------------------------------------
// This header provides convenient overrides for the new and
// delete operations in C++.
//
// This header should be included in only one source file! 
// 
// See <https://en.cppreference.com/w/cpp/memory/new/operator_new>
// ---------------------------------------------------------------------------
#include "jemalloc_override.h"
#include <new>

#if __cplusplus > 202002L
#warning Only operator new/delete up to C++20 overridden. If later standards add additional overrides they should be added here.
#endif


// (new 1)
[[nodiscard]] void* operator new(std::size_t count) {
    void* result = malloc(count);
    if (result == nullptr) {
        throw std::bad_alloc();
    }
    return result;
}

// (new 2)
[[nodiscard]] void* operator new[](std::size_t count) {
    void* result = malloc(count);
    if (result == nullptr) {
        throw std::bad_alloc();
    }
    return result;
}

// (new 3)
[[nodiscard]] void* operator new(std::size_t count, std::align_val_t al) {
    void* result = aligned_alloc(static_cast<size_t>(al), count);
    if (result == nullptr) {
        throw std::bad_alloc();
    }
    return result;
}

// (new 4)
[[nodiscard]] void* operator new[](std::size_t count, std::align_val_t al) {
    void* result = aligned_alloc(static_cast<size_t>(al), count);
    if (result == nullptr) {
        throw std::bad_alloc();
    }
    return result;
}

// (new 5)
[[nodiscard]] void* operator new(std::size_t count,
                                 const std::nothrow_t& tag) noexcept {
    void* result = malloc(count);
    return result;
}

// (new 6)
[[nodiscard]] void* operator new[](std::size_t count,
                                   const std::nothrow_t& tag) noexcept {
    void* result = malloc(count);
    return result;
}

// (new 7)
[[nodiscard]] void* operator new(std::size_t count,
                                 std::align_val_t al,
                                 const std::nothrow_t& tag) noexcept {
    void* result = aligned_alloc(static_cast<size_t>(al), count);
    return result;
}

// (new 8)
[[nodiscard]] void* operator new[](std::size_t count,
                                   std::align_val_t al,
                                   const std::nothrow_t& tag) noexcept {
    void* result = aligned_alloc(static_cast<size_t>(al), count);
    return result;
}

// (del 1)
void operator delete(void* ptr) noexcept {
    free(ptr);
}

// (del 2)
void operator delete[](void* ptr) noexcept {
    free(ptr);
}

// (del 3)
void operator delete(void* ptr, std::align_val_t) noexcept {
    free(ptr);
}

// (del 4)
void operator delete[](void* ptr, std::align_val_t) noexcept {
    free(ptr);
}

// (del 6)
void operator delete[](void* ptr, std::size_t size) noexcept {
    free(ptr);
}

// (del 5)
void operator delete(void* ptr, std::size_t size) noexcept {
    free(ptr);
}

// (del 7)
void operator delete(void* ptr, std::size_t size, std::align_val_t) noexcept {
    free(ptr);
}

// (del 8)
void operator delete[](void* ptr, std::size_t size, std::align_val_t) noexcept {
    free(ptr);
}

// (del 9)
void operator delete(void* ptr, const std::nothrow_t& tag) noexcept {
    free(ptr);
}

// (del 10)
void operator delete[](void* ptr, const std::nothrow_t& tag) noexcept {
    free(ptr);
}

#endif // USE_JEMALLOC
