/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#include "IndexBinaryIVF_c_ex.h"
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/clone_index.h>
#include <cstring>
#include "macros_impl.h"

using faiss::IndexBinaryIVF;
using faiss::SearchParameters;
using faiss::SearchParametersIVF;
using faiss::DirectMap;

extern "C" {
int faiss_IndexBinaryIVF_set_direct_map(
        FaissIndexBinaryIVF* index,
        int direct_map_type) {
    try {
        reinterpret_cast<IndexBinaryIVF*>(index)->set_direct_map_type(
                static_cast<DirectMap::Type>(direct_map_type));
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinaryIVF_list_vector_count(
        const FaissIndexBinaryIVF* index,
        idx_t* list_counts,
        size_t list_counts_size,
        const FaissSearchParametersIVF* params) {
    try {
        reinterpret_cast<const IndexBinaryIVF*>(index)->list_vector_count(
                list_counts,
                list_counts_size,
                reinterpret_cast<const SearchParameters*>(params));

    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinaryIVF_search_closest_eligible_centroids(
        const FaissIndexBinaryIVF* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* centroid_distances,
        idx_t* centroid_ids,
        const FaissSearchParameters* params) {
    try {
        reinterpret_cast<const IndexBinaryIVF*>(index)->quantizer->search(
                n,
                x,
                k,
                centroid_distances,
                centroid_ids,
                reinterpret_cast<const SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinaryIVF_search_preassigned_with_params(
        const FaissIndexBinaryIVF* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        const idx_t* assign,
        const int32_t* centroid_dis,
        int32_t* distances,
        idx_t* labels,
        int store_pairs,
        const FaissSearchParametersIVF* params) {
    try {
        reinterpret_cast<const IndexBinaryIVF*>(index)->search_preassigned(
                n,
                x,
                k,
                assign,
                centroid_dis,
                distances,
                labels,
                store_pairs,
                reinterpret_cast<const SearchParametersIVF*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinaryIVF_get_centroids_and_cardinality(
        const FaissIndexBinaryIVF* index,
        uint8_t* centroid_vectors,
        size_t* cardinalities,
        idx_t* centroid_ids) {
    try {
        reinterpret_cast<const IndexBinaryIVF*>(index)->get_centroids_and_cardinality(
            centroid_vectors, cardinalities, centroid_ids);
    }
    CATCH_AND_HANDLE
}

int faiss_Set_quantizers_binary(FaissIndexBinary* target, FaissIndexBinary* source) {
    try {
        auto* tgt = reinterpret_cast<faiss::IndexBinary*>(target);
        auto* src = reinterpret_cast<faiss::IndexBinary*>(source);
        assert(tgt && src);

        if (auto* tgt_bivf = dynamic_cast<faiss::IndexBinaryIVF*>(tgt)) {
            auto* src_bivf = dynamic_cast<faiss::IndexBinaryIVF*>(src);
            assert(src_bivf);

            tgt_bivf->quantizer = clone_binary_index(src_bivf->quantizer);
            tgt_bivf->is_trained = true;
            return 0;
        }
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinaryIVF_same_coarse_quantizer(
        const FaissIndexBinaryIVF* a,
        const FaissIndexBinaryIVF* b,
        int* out) {
    try {
        FAISS_THROW_IF_NOT_MSG(out, "output pointer must not be null");
        *out = 0;

        const auto* ivf_a = reinterpret_cast<const IndexBinaryIVF*>(a);
        const auto* ivf_b = reinterpret_cast<const IndexBinaryIVF*>(b);
        if (ivf_a == nullptr || ivf_b == nullptr) {
            return 0;
        }
        if (ivf_a == ivf_b || ivf_a->quantizer == ivf_b->quantizer) {
            *out = 1;
            return 0;
        }
        if (ivf_a->d != ivf_b->d || ivf_a->nlist != ivf_b->nlist) {
            return 0;
        }
        if (ivf_a->quantizer == nullptr || ivf_b->quantizer == nullptr) {
            return 0;
        }
        if (ivf_a->quantizer->ntotal != ivf_b->quantizer->ntotal ||
            ivf_a->quantizer->ntotal != (faiss::idx_t)ivf_a->nlist) {
            return 0;
        }

        const auto* flat_a =
                dynamic_cast<const faiss::IndexBinaryFlat*>(ivf_a->quantizer);
        const auto* flat_b =
                dynamic_cast<const faiss::IndexBinaryFlat*>(ivf_b->quantizer);
        if (flat_a == nullptr || flat_b == nullptr) {
            return 0;
        }
        // IndexBinaryFlat stores d bits per vector, i.e. d / 8 bytes.
        const size_t nbytes = (size_t)(ivf_a->d / 8) * (size_t)ivf_a->nlist;
        if (nbytes == 0) {
            return 0;
        }
        const uint8_t* codes_a = flat_a->xb.data();
        const uint8_t* codes_b = flat_b->xb.data();
        if (codes_a == nullptr || codes_b == nullptr) {
            return 0;
        }
        if (codes_a == codes_b || memcmp(codes_a, codes_b, nbytes) == 0) {
            *out = 1;
        }
        return 0;
    }
    CATCH_AND_HANDLE
}

}
