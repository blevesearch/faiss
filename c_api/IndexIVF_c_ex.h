/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2023-present Facebook. All Rights Reserved.
// -*- c -*-

#ifndef FAISS_INDEX_IVF_EX_C_H
#define FAISS_INDEX_IVF_EX_C_H

#include "Clustering_c.h"
#include "Index_c.h"
#include "faiss_c.h"
#include "IndexIVF_c.h"

#ifdef __cplusplus
extern "C" {
#endif
int faiss_IndexIVF_set_direct_map(
        FaissIndexIVF* index,
        int direct_map_type);

int faiss_SearchParametersIVF_new_with_sel(
        FaissSearchParametersIVF** p_sp,
        FaissIDSelector* sel);
/*
    Return 'k' centroids in the index closest to the query vector.

    @param n: number of queries.
    @param query: query vector.
    @param k: count of closest number of vectors.
    @param centroid_distances: output distances, size n * k.
    @param centroid_ids: output centroid IDs, size n * k.    
*/
int faiss_Search_closest_eligible_centroids(
        FaissIndex* index,
        int n,
        float* query,
        int k,
        float* centroid_distances,
        idx_t* centroid_ids
);

/*
    Search the clusters whose IDs are in 'assign' and
    return return the 'k' nearest neighbours from among them.     

    @param n: number of queries.
    @param x: query vector, size n * d.
    @param k: count of nearest neighbours to be returned for each query.
    @param centroid_ids: output centroid IDs, size n * k. 
    @param distance: output distances, size n * k
    @param labels: output labels, size n * k
*/
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
        const FaissSearchParametersIVF* params);

    /** Given a query vector x, compute distance to provided codes
     * for the input list_no. This is a special purpose method
     * to be used as a flat distance computer for an inverted
     * list where codes are provided externally. This allows to
     * use the quantizer while computing distance for the quantized
     * codes.
     *
     * @param list_no list number for inverted list
     * @param x - input query vector
     * @param n - number of codes
     * @param codes - input codes
     * @param dists - output computed distances
     */

int faiss_IndexIVF_compute_distance_to_codes_for_list(
        FaissIndexIVF* index,
        idx_t list_no,
        const float* x,
        idx_t n,
        const uint8_t* codes,
        float* dists);


#ifdef __cplusplus
}
#endif

#endif
