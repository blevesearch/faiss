/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2023-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "Index_c_ex.h"
#include <faiss/Index.h>
#include "macros_impl.h"

extern "C" {

int faiss_Index_reconstruct_batch(
        const FaissIndex* index,
        idx_t n,
        const idx_t* keys,
        float* recons) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->reconstruct_batch(
                n, keys, recons);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_merge_from(
        FaissIndex* index,
        FaissIndex* other,
        const idx_t add_id) {
    try {
        reinterpret_cast<faiss::Index*>(index)->merge_from(
                *reinterpret_cast<faiss::Index*>(other), add_id);
    }
    CATCH_AND_HANDLE
}

size_t faiss_Index_size(FaissIndex* index) {
    auto xIndex = reinterpret_cast<faiss::Index*>(index);
    size_t rv = sizeof(xIndex);
    return rv;
}
}