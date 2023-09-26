/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//  Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-
// I/O code for indexes

#include "index_io_c.h"
#include <faiss/index_io.h>
#include <chrono>
#include <faiss/impl/io.h>
#include "macros_impl.h"
using namespace std::chrono_literals;

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

using faiss::Index;
using faiss::IndexBinary;

int faiss_write_index(const FaissIndex* idx, FILE* f) {
    try {
        faiss::write_index(reinterpret_cast<const Index*>(idx), f);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_fname(const FaissIndex* idx, const char* fname) {
    try {
        faiss::write_index(reinterpret_cast<const Index*>(idx), fname);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index(FILE* f, int io_flags, FaissIndex** p_out) {
    try {
        auto out = faiss::read_index(f, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_fname(
        const char* fname,
        int io_flags,
        FaissIndex** p_out) {
    try {
        auto out = faiss::read_index(fname, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary(const FaissIndexBinary* idx, FILE* f) {
    try {
        faiss::write_index_binary(reinterpret_cast<const IndexBinary*>(idx), f);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary_fname(
        const FaissIndexBinary* idx,
        const char* fname) {
    try {
        faiss::write_index_binary(
                reinterpret_cast<const IndexBinary*>(idx), fname);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary(FILE* f, int io_flags, FaissIndexBinary** p_out) {
    try {
        auto out = faiss::read_index_binary(f, io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary_fname(
        const char* fname,
        int io_flags,
        FaissIndexBinary** p_out) {
    try {
        auto out = faiss::read_index_binary(fname, io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}
