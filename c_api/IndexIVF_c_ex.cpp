/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2023-present Facebook. All Rights Reserved.
// -*- c++ -*-
#include "IndexIVF_c_ex.h"
#include <faiss/IndexIVF.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/clone_index.h>
#include "macros_impl.h"

using faiss::IndexIVF;
using faiss::SearchParameters;
using faiss::SearchParametersIVF;

int faiss_IndexIVF_set_direct_map(FaissIndexIVF* index, int direct_map_type) {
    try {
        reinterpret_cast<IndexIVF*>(index)->set_direct_map_type(
                static_cast<faiss::DirectMap::Type>(direct_map_type));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_list_vector_count(
        const FaissIndexIVF* index,
        idx_t* list_counts,
        size_t list_counts_size,
        const FaissSearchParametersIVF* params) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->ivf_list_vector_count(
                list_counts,
                list_counts_size,
                reinterpret_cast<const faiss::SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_search_closest_eligible_centroids(
        const FaissIndexIVF* index,
        idx_t n,
        const float* query,
        idx_t k,
        float* centroid_distances,
        idx_t* centroid_ids,
        const FaissSearchParameters* params) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->quantizer->search(
                n,
                query,
                k,
                centroid_distances,
                centroid_ids,
                reinterpret_cast<const faiss::SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_Set_quantizers(FaissIndex* target, FaissIndex* source) {
    try {
        auto* tgt = reinterpret_cast<faiss::Index*>(target);
        auto* src = reinterpret_cast<faiss::Index*>(source);

        assert(tgt && src);

        // -------- IndexIVFScalarQuantizer --------
        if (auto* tgt_ivfsq = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(tgt)) {
            auto* src_ivfsq = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(src);
            assert(src_ivfsq);

            tgt_ivfsq->quantizer = src_ivfsq->quantizer;
            tgt_ivfsq->is_trained = true;
            tgt_ivfsq->sq = src_ivfsq->sq;
            return 0;
        }

        // -------- IndexIVFRaBitQ --------
        if (auto* tgt_rabitq = dynamic_cast<faiss::IndexIVFRaBitQ*>(tgt)) {
            auto* src_rabitq = dynamic_cast<faiss::IndexIVFRaBitQ*>(src);
            assert(src_rabitq);

            tgt_rabitq->quantizer = src_rabitq->quantizer;
            tgt_rabitq->is_trained = true;
            tgt_rabitq->rabitq = src_rabitq->rabitq;
            return 0;
        }

        // Unsupported type
        return -1;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_search_preassigned_with_params(
        const FaissIndexIVF* index,
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        int store_pairs,
        const FaissSearchParametersIVF* params) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->search_preassigned(
                n,
                x,
                k,
                assign,
                centroid_dis,
                distances,
                labels,
                store_pairs,
                reinterpret_cast<const faiss::SearchParametersIVF*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_compute_distance_to_codes_for_list(
        FaissIndexIVF* index,
        idx_t list_no,
        const float* x,
        idx_t n,
        const uint8_t* codes,
        float* dists,
        float* dist_table) {
    try {
        reinterpret_cast<IndexIVF*>(index)->compute_distance_to_codes_for_list(
                list_no, x, n, codes, dists, dist_table);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_get_centroids_and_cardinality(
        const FaissIndexIVF* index,
        float* centroid_vectors,
        size_t* cardinalities,
        idx_t* centroid_ids) {
    try {
        reinterpret_cast<const IndexIVF*>(index)->get_centroids_and_cardinality(
            centroid_vectors, cardinalities, centroid_ids);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_compute_distance_table(
        FaissIndexIVF* index,
        const float* x,
        float* dist_table) {
    try {
        reinterpret_cast<IndexIVF*>(index)->compute_distance_table(
                x, dist_table);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVF_has_RaBitQ(FaissIndex* index) {
    try {
        faiss::Index* idx = reinterpret_cast<faiss::Index*>(index);

        faiss::IndexIVFRaBitQ* ivf_rq =
            dynamic_cast<faiss::IndexIVFRaBitQ*>(idx);

        if (ivf_rq) {
            return 0;
        }

        return -1;
    }
    CATCH_AND_HANDLE
}

int faiss_SearchParametersRaBitQ_new_with(
        FaissSearchParametersIVF** p_sp,
        FaissIDSelector* sel,
        size_t nprobe,
        size_t max_codes) {
    try {
        faiss::IVFRaBitQSearchParameters* rqsp = new faiss::IVFRaBitQSearchParameters;
        rqsp->centered = true;
        rqsp->qb = 4;
        rqsp->sel = reinterpret_cast<faiss::IDSelector*>(sel);
        rqsp->nprobe = nprobe;
        rqsp->max_codes = max_codes;

        *p_sp = reinterpret_cast<FaissSearchParametersIVF*>(rqsp);
        return 0;
    }
    CATCH_AND_HANDLE
}