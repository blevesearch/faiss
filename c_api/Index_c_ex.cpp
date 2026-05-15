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
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFRaBitQ.h>

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

int faiss_Index_size(const FaissIndex* index, size_t* p_size) {
    try {
        const faiss::Index* idx = reinterpret_cast<const faiss::Index*>(index);
        // Base: raw vector codes (works for Flat, SQ, and all other types).
        size_t size = (size_t)idx->ntotal * idx->sa_code_size();
        // Static struct footprint
        size += faiss_index_static_size(idx);
        // IVF-specific overhead not captured by sa_code_size():
        //   centroids: quantizer->ntotal * quantizer->sa_code_size()
        //   stored IDs: ntotal * sizeof(idx_t)  (per-vector ID in each inverted list)
        //   quantizer struct footprint
        if (auto ivf = dynamic_cast<const faiss::IndexIVF*>(idx)) {
            auto ivfQuantizer = ivf->quantizer;
            if (ivfQuantizer != nullptr) {
                size += (size_t)ivfQuantizer->ntotal * ivfQuantizer->sa_code_size();
                size += faiss_index_static_size(ivfQuantizer);
            }
            size += (size_t)ivf->ntotal * sizeof(faiss::idx_t);
        }
        *p_size = size;
    }
    CATCH_AND_HANDLE
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

static size_t faiss_index_static_size(const faiss::Index* idx) {
    if (idx == nullptr) {
        return 0;
    }
    // Flat Index
    if (dynamic_cast<const faiss::IndexFlat*>(idx)) {
        return sizeof(faiss::IndexFlat);
    }
    // SQ Index
    if (dynamic_cast<const faiss::IndexScalarQuantizer*>(idx)) {
        return sizeof(faiss::IndexScalarQuantizer);
    }
    // IVF,SQ Index
    if (dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(idx)) {
        return sizeof(faiss::IndexIVFScalarQuantizer);
    }
    // IVF,Flat Index
    if (dynamic_cast<const faiss::IndexIVFFlat*>(idx)) {
        return sizeof(faiss::IndexIVFFlat);
    }
    // IVF,RaBitQ Index
    if (dynamic_cast<const faiss::IndexIVFRaBitQ*>(idx)) {
        return sizeof(faiss::IndexIVFRaBitQ);
    }
    // IVF Index
    if (dynamic_cast<const faiss::IndexIVF*>(idx)) {
        return sizeof(faiss::IndexIVF);
    }
    // Base Index
    return sizeof(faiss::Index);
}

int faiss_Index_static_size(const FaissIndex* index, size_t* p_size) {
    try {
        const faiss::Index* idx = reinterpret_cast<const faiss::Index*>(index);
        size_t size = faiss_index_static_size(idx);
        // For IVF indices, include quantizer struct footprint
        if (auto ivf = dynamic_cast<const faiss::IndexIVF*>(idx)) {
            if (ivf->quantizer != nullptr) {
                size += faiss_index_static_size(ivf->quantizer);
            }
        }
        *p_size = size;
    }
    CATCH_AND_HANDLE
}

}
