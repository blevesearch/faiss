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
#include <faiss/index_io.h>
#include <faiss/impl/io.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexBinary;

int faiss_write_index_buf(const FaissIndex* idx, int* size, unsigned char** buf) {
    try {
        faiss::VectorIOWriter writer;

        faiss::write_index(reinterpret_cast<const Index*>(idx), &writer);
        unsigned char* tempBuf = (unsigned char*)malloc((writer.data.size()) * sizeof(uint8_t));
        std::copy(writer.data.begin(), writer.data.end(), tempBuf);
        *buf = tempBuf;
        *size = writer.data.size();
        writer.data.clear();
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_buf(const unsigned char* buf, int size, int io_flags, FaissIndex** p_out) {
    try {
        faiss::VectorIOReader reader;
        reader.data.assign(buf, buf + size);
        auto index = faiss::read_index(&reader, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(index);
    }
    CATCH_AND_HANDLE
}
