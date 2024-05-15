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
// already mmap'd data buffer. it's C equivalent of IO_FLAG_READ_MMAP from index_io.h
// usage is - FAISS_IO_FLAG_READ_MMAP | FAISS_IO_FLAG_ONDISK_IVF
#define FAISS_IO_FLAG_READ_MMAP 64
#define FAISS_IO_FLAG_ONDISK_IVF 0x646f0000

/** Write index to buffer
 */
int faiss_write_index_buf(const FaissIndex* idx, size_t* buf_size, unsigned char** buf);

/** Read index from buffer
 */
int faiss_read_index_buf(const unsigned char* buf, size_t limit, int io_flags,
        FaissIndex** p_out);

#ifdef __cplusplus
}
#endif
#endif
