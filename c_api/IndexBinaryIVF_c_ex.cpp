/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#include "IndexBinaryIVF_c_ex.h"
#include <faiss/IndexBinaryIVF.h>
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
}
