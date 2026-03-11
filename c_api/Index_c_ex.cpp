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
#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>

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

int faiss_Index_dist_compute(
        const FaissIndex* index,
        const float* query,
        const idx_t* ids,
        size_t n_ids,
        float* distances) {
    try {
        const faiss::Index* idx = reinterpret_cast<const faiss::Index*>(index);

        // Try to cast to IndexFlat
        if (auto flat = dynamic_cast<const faiss::IndexFlat*>(idx)) {
            // For IndexFlat, we can use compute_distance_subset
            flat->compute_distance_subset(1, query, n_ids, distances, ids);
            return 0;
        }

        // Try to cast to IndexScalarQuantizer
        if (auto isq =
                    dynamic_cast<const faiss::IndexScalarQuantizer*>(idx)) {
            isq->dist_compute(query, ids, n_ids, distances);
            return 0;
        }

        // If we get here, the index type doesn't support dist_compute
        return -1;
    }
    CATCH_AND_HANDLE
}
}
