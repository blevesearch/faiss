/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//  Copyright 2023-present Facebook. All Rights Reserved
// -*- c++ -*-
// More I/O code for indexes

#include "index_io_c_ex.h"
#include <cstring>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexBinary;

namespace {

/** Counts bytes and stores nothing. Used to size a buffer before filling it. */
struct CountingIOWriter : faiss::IOWriter {
    size_t total = 0;

    size_t operator()(const void*, size_t size, size_t nitems) override {
        total += size * nitems;
        return nitems;
    }
};

/** Writes into a fixed caller-provided buffer, refusing to overrun it. */
struct FixedBufIOWriter : faiss::IOWriter {
    uint8_t* buf;
    size_t capacity;
    size_t pos = 0;

    FixedBufIOWriter(uint8_t* b, size_t cap) : buf(b), capacity(cap) {
        name = "fixed buffer";
    }

    size_t operator()(const void* ptr, size_t size, size_t nitems) override {
        const size_t n = size * nitems;
        FAISS_THROW_IF_NOT_FMT(
                pos + n <= capacity,
                "serialized index does not fit: %zu bytes written, %zu more needed, buffer holds %zu",
                pos,
                n,
                capacity);
        memcpy(buf + pos, ptr, n);
        pos += n;
        return nitems;
    }
};

} // namespace

int faiss_index_serialized_size(const FaissIndex* idx, size_t* size) {
    try {
        CountingIOWriter counter;
        faiss::write_index(reinterpret_cast<const Index*>(idx), &counter);
        *size = counter.total;
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_into_buf(
        const FaissIndex* idx,
        uint8_t* buf,
        size_t capacity,
        size_t* written) {
    try {
        FAISS_THROW_IF_NOT_MSG(
                buf != nullptr || capacity == 0,
                "null buffer passed with a non-zero capacity");
        FixedBufIOWriter writer(buf, capacity);
        faiss::write_index(reinterpret_cast<const Index*>(idx), &writer);
        *written = writer.pos;
    }
    CATCH_AND_HANDLE
}

int faiss_index_binary_serialized_size(
        const FaissIndexBinary* idx,
        size_t* size) {
    try {
        CountingIOWriter counter;
        faiss::write_index_binary(
                reinterpret_cast<const IndexBinary*>(idx), &counter);
        *size = counter.total;
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary_into_buf(
        const FaissIndexBinary* idx,
        uint8_t* buf,
        size_t capacity,
        size_t* written) {
    try {
        FAISS_THROW_IF_NOT_MSG(
                buf != nullptr || capacity == 0,
                "null buffer passed with a non-zero capacity");
        FixedBufIOWriter writer(buf, capacity);
        faiss::write_index_binary(
                reinterpret_cast<const IndexBinary*>(idx), &writer);
        *written = writer.pos;
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_buf(const FaissIndex* idx, size_t* size, uint8_t** buf) {
    try {
        faiss::VectorIOWriter writer;
        faiss::write_index(reinterpret_cast<const Index*>(idx), &writer);
        uint8_t* tempBuf =
                (uint8_t*)malloc((writer.data.size()) * sizeof(uint8_t));
        std::copy(writer.data.begin(), writer.data.end(), tempBuf);
        *buf = tempBuf;
        *size = writer.data.size();
        writer.data.clear();
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_buf(
        const uint8_t* buf,
        size_t size,
        int io_flags,
        FaissIndex** p_out) {
    try {
        faiss::BufIOReader reader;
        reader.buf = buf;
        reader.buf_size = size;
        auto index = faiss::read_index(&reader, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(index);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary_buf(const FaissIndexBinary* idx, size_t* size, uint8_t** buf) {
    try {
        faiss::VectorIOWriter writer;
        faiss::write_index_binary(reinterpret_cast<const IndexBinary*>(idx), &writer);
        uint8_t* tempBuf = (uint8_t*)malloc((writer.data.size()) * sizeof(uint8_t));
        std::copy(writer.data.begin(), writer.data.end(), tempBuf);
        *buf = tempBuf;
        *size = writer.data.size();
        writer.data.clear();
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary_buf(const uint8_t* buf, size_t size, int io_flags, FaissIndexBinary** p_out) {
    try {
        faiss::BufIOReader reader;
        reader.buf = buf;
        reader.buf_size = size;
        auto index = faiss::read_index_binary(&reader, io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(index);
    }
    CATCH_AND_HANDLE
}


void faiss_free_buf(uint8_t** buf) {
    free(*buf);
    *buf = nullptr;
}
