/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c -*-

#include "IndexBinary_c_ex.h"
#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include "macros_impl.h"

extern "C" {
int faiss_IndexBinary_search_with_params(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        const FaissSearchParameters* params,
        int32_t* distances,
        idx_t* labels) {
    try {
        reinterpret_cast<const faiss::IndexBinary*>(index)->search(
                n,
                x,
                k,
                distances,
                labels,
                reinterpret_cast<const faiss::SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_size(const FaissIndexBinary* index, size_t* p_size) {
    try {
        const faiss::IndexBinary* idx = reinterpret_cast<const faiss::IndexBinary*>(index);
        // Base: raw binary codes (d / 8 bytes per vector).
        size_t size = (size_t)idx->ntotal * idx->code_size;
        // Static struct footprint
        size += faiss_index_binary_static_size(idx);
        // IVF-specific overhead not captured by code_size:
        //   centroids: quantizer->ntotal * quantizer->sa_code_size()
        //   stored IDs: ntotal * sizeof(idx_t)  (per-vector ID in each inverted list)
        //   quantizer struct footprint
        if (auto ivf = dynamic_cast<const faiss::IndexBinaryIVF*>(idx)) {
            auto ivfQuantizer = ivf->quantizer;
            if (ivfQuantizer != nullptr) {
                size += (size_t)ivfQuantizer->ntotal * ivfQuantizer->sa_code_size();
                size += faiss_index_binary_static_size(ivfQuantizer);
            }
            size += (size_t)ivf->ntotal * sizeof(faiss::idx_t);
        }
        *p_size = size;
    }
    CATCH_AND_HANDLE
}

static size_t faiss_index_binary_static_size(const faiss::IndexBinary* idx) {
    if (idx == nullptr) {
        return 0;
    }
    // BFlat
    if (dynamic_cast<const faiss::IndexBinaryFlat*>(idx)) {
        return sizeof(faiss::IndexBinaryFlat);
    }
    // BIVF
    if (dynamic_cast<const faiss::IndexBinaryIVF*>(idx)) {
        return sizeof(faiss::IndexBinaryIVF);
    }
    // Base
    return sizeof(faiss::IndexBinary);
}

int faiss_IndexBinary_static_size(const FaissIndexBinary* index, size_t* p_size) {
    try {
        const faiss::IndexBinary* idx = reinterpret_cast<const faiss::IndexBinary*>(index);
        size_t size = faiss_index_binary_static_size(idx);
        // For IVF indices, include quantizer struct footprint
        if (auto ivf = dynamic_cast<const faiss::IndexBinaryIVF*>(idx)) {
            if (ivf->quantizer != nullptr) {
                size += faiss_index_binary_static_size(ivf->quantizer);
            }
        }
        *p_size = size;
    }
    CATCH_AND_HANDLE
}
}
