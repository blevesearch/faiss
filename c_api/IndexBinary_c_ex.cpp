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

int faiss_IndexBinary_size(FaissIndexBinary* index, size_t* p_size) {
    try {
        const faiss::IndexBinary* idx = reinterpret_cast<const faiss::IndexBinary*>(index);

        // Base: raw binary codes (d / 8 bytes per vector).
        size_t size = (size_t)idx->ntotal * idx->code_size;

        // IVF-specific overhead not captured by code_size:
        //   centroids: quantizer->ntotal * quantizer->sa_code_size()
        //   stored IDs: ntotal * sizeof(idx_t)  (per-vector ID in each inverted list)
        if (auto ivf = dynamic_cast<const faiss::IndexBinaryIVF*>(idx)) {
            auto ivfQuantizer = ivf->quantizer;
            size += (size_t)ivfQuantizer->ntotal * ivfQuantizer->sa_code_size();
            size += (size_t)ivf->ntotal * sizeof(faiss::idx_t);
        }

        *p_size = size;
        return 0;
    }
    CATCH_AND_HANDLE
}
}
