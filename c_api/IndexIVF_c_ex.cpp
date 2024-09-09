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
#include "macros_impl.h"

using faiss::IndexIVF;
using faiss::SearchParametersIVF;

int faiss_IndexIVF_set_direct_map(FaissIndexIVF* index, int direct_map_type) {
    try {
        reinterpret_cast<IndexIVF*>(index)->set_direct_map_type(
                static_cast<faiss::DirectMap::Type>(direct_map_type));
    }
    CATCH_AND_HANDLE
}

int faiss_SearchParametersIVF_new_with_sel(
        FaissSearchParametersIVF** p_sp,
        FaissIDSelector* sel) {
    try {
        SearchParametersIVF* sp = new SearchParametersIVF;
        sp->sel = reinterpret_cast<faiss::IDSelector*>(sel);
        *p_sp = reinterpret_cast<FaissSearchParametersIVF*>(sp);
    }
    CATCH_AND_HANDLE
}

int faiss_Search_closest_eligible_centroids(
        FaissIndex* index,
        float* query,
        int n, // top n centroids.
        float* centroid_distances,
        idx_t* centroid_ids) {
    try {
        faiss::IndexIVF* index_ivf = reinterpret_cast<IndexIVF*>(index); 
        assert(index_ivf);

        index_ivf->quantizer->search(1, query, n, centroid_distances, centroid_ids);
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
                n, x, k, assign, centroid_dis, distances, labels, store_pairs,
                reinterpret_cast<const faiss::SearchParametersIVF*>(params));
    }
    CATCH_AND_HANDLE
}