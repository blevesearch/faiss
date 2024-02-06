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

/** Write index to buffer
 */
int faiss_write_index_buf(const FaissIndex* idx, size_t* buf_size, unsigned char** buf);

/** Read index from buffer
 */
int faiss_read_index_buf(const unsigned char* buf, int limit, int io_flags,
        FaissIndex** p_out);

#ifdef __cplusplus
}
#endif
#endif
