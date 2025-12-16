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
#include "IndexIVF_c.h"
#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif
int faiss_IndexIVF_set_direct_map(FaissIndexIVF* index, int direct_map_type);

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
        idx_t n,
        const float* query,
        idx_t k,
        float* centroid_distances,
        idx_t* centroid_ids,
        const FaissSearchParameters* params);

/*
    Search the clusters whose IDs are in 'assign' and
    return the 'k' nearest neighbours from among them.

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

/*
    Given a query vector x, compute distance to provided codes
    for the input list_no. This is a special purpose method
    to be used as a flat distance computer for an inverted
    list where codes are provided externally. This allows to
    use the quantizer independently while computing distance
    for the quantized codes.

    @param list_no list number for inverted list
    @param x - input query vector
    @param n - number of codes
    @param codes - input codes
    @param dists - output computed distances
    @param dist_table - input precomputed distance table for PQ
*/

int faiss_IndexIVF_compute_distance_to_codes_for_list(
        FaissIndexIVF* index,
        idx_t list_no,
        const float* x,
        idx_t n,
        const uint8_t* codes,
        float* dists,
        float* dist_table);

/*
    Given multiple vector IDs, retrieve the corresponding list (cluster) IDs
    from an IVF index. This function efficiently assigns vector IDs to their
    respective inverted lists/clusters in a batch operation.

    @param index  - Pointer to the Faiss IVF index
    @param keys   - Input array of vector IDs (keys)
    @param n_keys - Number of vector keys in the input array
    @param lists  - Output array where corresponding cluster (list) IDs are stored
*/

int faiss_get_lists_for_keys(
        FaissIndexIVF* index,
        idx_t* keys,
        size_t n_keys,
        idx_t* lists);

/*
    Get centroid information and cardinality for all centroids in an IVF index.

    @param index: the IVF index
    @param centroid_vectors: output array for centroid vectors, size nlist * d
    @param cardinalities: output array for cardinalities, size nlist
    @param centroid_ids: output array for centroid IDs, size nlist (optional, can be nullptr)
*/
int faiss_IndexIVF_get_centroids_and_cardinality(
        const FaissIndexIVF* index,
        float* centroid_vectors,
        size_t* cardinalities,
        idx_t* centroid_ids);

/*
    Given a query vector x, compute distance table and
    return to the caller.

    @param x - input query vector
    @param dist_table - output precomputed distance table for PQ

*/

int faiss_IndexIVF_compute_distance_table(
        FaissIndexIVF* index,
        const float* x,
        float* dist_table);

#ifdef __cplusplus
}
#endif

#endif
