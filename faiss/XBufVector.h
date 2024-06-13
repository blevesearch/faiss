#ifndef X_BUF_VECTOR_H
#define X_BUF_VECTOR_H

#include <cassert>
#include <sys/types.h>

namespace faiss {

// A XBufVector<T> is an alternative to std::vector<T> and is meant to
// implement the bare-minimum subset of std::vector<T> API as used by
// faiss, where XBufVector<T> instead points to some underlying
// read-only buffer_ of contiguous T items.
//
// The read-only buffer_, for example, might be memory that came
// from mmap()'s PROT_READ + MAP_SHARED.
//
// For mutation situations, XBufVector<T> will on-demand copy its
// read-only buffer_ of T items (if any) into a newly allocated
// std::vector<T> instance v_ in order to support any further
// mutation methods -- see the mutate() method.
template <typename T>
struct XBufVector {
    T* buffer_;
    std::size_t size_;
    std::size_t capacity_;

    // Will be NULL when we're using the read-only buffer_
    // and non-NULL when we've been mutate()'ed and will
    // forward all method calls to v_.
    std::vector<T> *v_;

    XBufVector()
        : buffer_(nullptr), size_(0), capacity_(0), v_(nullptr) {}

    XBufVector(T* buffer, std::size_t size, std::size_t capacity)
        : buffer_(buffer), size_(size), capacity_(capacity), v_(nullptr) {}

    XBufVector(std::vector<T> *v)
        : buffer_(nullptr), size_(0), capacity_(0), v_(v) {}

    ~XBufVector() {
        // The buffer_ memory is not owned by us,
        // but the v_ vector is owned by us.
        if (v_ != nullptr) {
            delete v_;
        }
    }

    void mutate() {
        if (v_ == nullptr) {
            v_ = new std::vector<T>(size_);

            // Copy items from our read-only buffer_ into v_ so
            // that we can handle mutation methods.
            for (size_t i = 0; i < size_; i++) {
                v_->at(i) = buffer_[i];
            }
        }

        buffer_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    void setBuffer(uint8_t* buffer, std::size_t size, std::size_t capacity) {
        if (v_ != nullptr)
            throw std::out_of_range("XBufVector setBuffer() already has non-NULL v_");

        buffer_ = (T*) buffer;
        size_ = size;
        capacity_ = capacity;
    }

    std::size_t size() const {
        if (v_ != nullptr) return v_->size();
        return size_;
    }
    std::size_t capacity() const {
        if (v_ != nullptr) return v_->capacity();
        return capacity_;
    }
    bool empty() const {
        if (v_ != nullptr) return v_->empty();
        return size_ == 0;
    }

    void resize(size_t n, T val = T()) {
        mutate();
        v_->resize(n, val);
    }

    void swap(std::vector<T> &other) {
        mutate();
        v_->swap(other);
    }

    T& operator[](std::size_t index) {
        // TODO: We can't tell if this a read-only access or mutational?
        if (v_ != nullptr) return v_->at(index);
        assert(index < size_);
        return buffer_[index];
    }
    const T& operator[](std::size_t index) const {
        // TODO: We can't tell if this a read-only access or mutational?
        if (v_ != nullptr) return v_->at(index);
        assert(index < size_);
        return buffer_[index];
    }

    T& at(std::size_t index) {
        // TODO: We can't tell if this a read-only access or mutational?
        if (v_ != nullptr) return v_->at(index);
        if (index >= size_)
            throw std::out_of_range("XBufVector at index out of range");
        return buffer_[index];
    }
    const T& at(std::size_t index) const {
        // TODO: We can't tell if this a read-only access or mutational?
        if (v_ != nullptr) return v_->at(index);
        if (index >= size_)
            throw std::out_of_range("XBufVector const at index out of range");
        return buffer_[index];
    }

    T* data() {
        // TODO: We can't tell if this a read-only access or mutational?
        if (v_ != nullptr) return v_->data();
        return buffer_;
    }
    const T* data() const {
        // TODO: We can't tell if this a read-only access or mutational?
        if (v_ != nullptr) return v_->data();
        return buffer_;
    }

    T& front() {
        assert(size_ > 0);
        return buffer_[0];
    }
    const T& front() const {
        assert(size_ > 0);
        return buffer_[0];
    }

    T& back() {
        assert(size_ > 0);
        return buffer_[size_ - 1];
    }
    const T& back() const {
        assert(size_ > 0);
        return buffer_[size_ - 1];
    }

    void push_back(const T& value) {
        mutate();
        v_->push_back(value);
    }

    void pop_back() {
        mutate();
        v_->pop_back();
    }

    void clear() {
        if (v_ != nullptr) {
            v_->clear();
            v_ = nullptr;
            return;
        }

        buffer_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }
};

}

// ----------------------------------------------------------------------
// Variants of the READVECTOR() macros (see io_macros.h) to be
// used when vec is a XBufVector<T> and f is a BufIOReader.

#define X_READVECTOR(vec)                                                \
    {                                                                    \
        BufIOReader* reader = dynamic_cast<BufIOReader*>(f);             \
        if (reader != nullptr) {                                         \
            size_t size;                                                 \
            READANDCHECK(&size, 1);                                      \
            FAISS_THROW_IF_NOT(size >= 0 && size < (uint64_t{1} << 40)); \
            (vec).setBuffer(reader->readPointer(size), size, size);      \
        } else {                                                         \
            READVECTOR(vec);                                             \
        }                                                                \
    }


#define X_READXBVECTOR(vec)                                              \
    {                                                                    \
        BufIOReader* reader = dynamic_cast<BufIOReader*>(f);             \
        if (reader != nullptr) {                                         \
            size_t size;                                                 \
            READANDCHECK(&size, 1);                                      \
            FAISS_THROW_IF_NOT(size >= 0 && size < (uint64_t{1} << 40)); \
            size *= 4;                                                   \
            (vec).setBuffer(reader->readPointer(size), size, size);      \
        } else {                                                         \
            READXBVECTOR(vec);                                           \
        }                                                                \
    }

#endif // X_BUF_VECTOR_H
