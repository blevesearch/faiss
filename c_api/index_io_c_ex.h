/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//  Copyright 2023-present Facebook. All Rights Reserved
// -*- c++ -*-
// More I/O code for indexes

#ifndef FAISS_INDEX_IO_C_EX_H
#define FAISS_INDEX_IO_C_EX_H

#include <stdio.h>
#include "IndexBinary_c.h"
#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

// skip prefetch phase while searching over the inverted lists
#define FAISS_IO_FLAG_SKIP_PREFETCH 32
// the following two macros together decide whether to read the index from an
// already mmap'd data buffer. it's C equivalent of IO_FLAG_READ_MMAP from
// index_io.h usage is - FAISS_IO_FLAG_READ_MMAP | FAISS_IO_FLAG_ONDISK_IVF
#define FAISS_IO_FLAG_READ_MMAP 64
#define FAISS_IO_FLAG_ONDISK_IVF 0x646f0000

/** Write index to buffer
 */
int faiss_write_index_buf(
        const FaissIndex* idx,
        size_t* buf_size,
        uint8_t** buf);

/** Read index from buffer
 */
int faiss_read_index_buf(
        const uint8_t* buf,
        size_t limit,
        int io_flags,
        FaissIndex** p_out);

/** Write binary index from buffer
 */
int faiss_write_index_binary_buf(
        const FaissIndexBinary* idx,
        size_t* buf_size,
        uint8_t** buf);

/** Read binary index from buffer
 */
int faiss_read_index_binary_buf(
        const uint8_t* buf,
        size_t limit,
        int io_flags,
        FaissIndexBinary** p_out);

/** Compute the exact number of bytes write_index would emit for this index,
 * without allocating a buffer to hold them.
 *
 * Serialises the index through a writer that only counts, so this costs a
 * serialisation pass of CPU and no memory at all. Pair it with
 * faiss_write_index_into_buf to have the caller own the only copy of the
 * serialised bytes: ask the size, allocate, then fill.
 *
 * @param idx    index to measure
 * @param size   output, bytes that faiss_write_index_into_buf will write
 */
int faiss_index_serialized_size(const FaissIndex* idx, size_t* size);

/** Serialise an index directly into a caller-provided buffer.
 *
 * Writes nothing of its own: no std::vector, no malloc. The buffer must be at
 * least faiss_index_serialized_size bytes; a short buffer is an error rather
 * than an overflow.
 *
 * @param idx       index to serialise
 * @param buf       caller-owned destination, at least `capacity` bytes
 * @param capacity  size of buf
 * @param written   output, bytes actually written
 */
int faiss_write_index_into_buf(
        const FaissIndex* idx,
        uint8_t* buf,
        size_t capacity,
        size_t* written);

/** Compute the exact number of bytes write_index_binary would emit for this
 * index, without allocating a buffer to hold them.
 *
 * The binary twin of faiss_index_serialized_size. Binary indexes go through
 * the same three-copy path as float ones -- faiss_write_index_binary_buf
 * fills a std::vector and then mallocs a copy of it -- so they need the same
 * treatment.
 *
 * @param idx    binary index to measure
 * @param size   output, bytes faiss_write_index_binary_into_buf will write
 */
int faiss_index_binary_serialized_size(
        const FaissIndexBinary* idx,
        size_t* size);

/** Serialise a binary index directly into a caller-provided buffer.
 *
 * The binary twin of faiss_write_index_into_buf. The buffer must be at least
 * faiss_index_binary_serialized_size bytes; a short buffer is an error rather
 * than an overflow.
 *
 * @param idx       binary index to serialise
 * @param buf       caller-owned destination, at least `capacity` bytes
 * @param capacity  size of buf
 * @param written   output, bytes actually written
 */
int faiss_write_index_binary_into_buf(
        const FaissIndexBinary* idx,
        uint8_t* buf,
        size_t capacity,
        size_t* written);

void faiss_free_buf(uint8_t** buf);

#ifdef __cplusplus
}
#endif
#endif
