/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_BINARY_IVF_C_EX_H
#define FAISS_INDEX_BINARY_IVF_C_EX_H

#include "IndexBinaryIVF_c.h"

#ifdef __cplusplus
extern "C" {
#endif

int faiss_IndexBinaryIVF_set_direct_map(
        FaissIndexBinaryIVF* index,
        int direct_map_type);

/** return list of centroids with counts of eligible centroids
 * 
 * @param list_counts       output counts of eligible vectors in each centroid
 * @param list_counts_size  total number of centroids
*/
int faiss_IndexBinaryIVF_list_vector_count(
        const FaissIndexBinaryIVF* index,
        idx_t* list_counts,
        size_t list_counts_size,
        const FaissSearchParametersIVF* params);

/** return 'k' centroids in the index closest to the query vector.
 *
 * @param n                     number of queries
 * @param x                     query vector
 * @param k                     count of closest number of vectors
 * @param centroid_distances    output distances, size n * k
 * @param centroid_ids          output centroid IDs, size n * k
*/
int faiss_IndexBinaryIVF_search_closest_eligible_centroids(
        const FaissIndexBinaryIVF* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* centroid_distances,
        idx_t* centroid_ids,
        const FaissSearchParameters* params);

/** search the clusters whose IDs are in 'assign' and
 * return the 'k' nearest neighbours from among them.
 * 
 * @param n              number of queries
 * @param x              query vector, size n * d
 * @param k              count of nearest neighbours to be
 *                         returned for each query
 * @param centroid_ids   output centroid IDs, size n * k
 * @param distance       output distances, size n * k
 * @param labels         output labels, size n * k
*/
int faiss_IndexBinaryIVF_search_preassigned_with_params(
        const FaissIndexBinaryIVF* index,
        idx_t n,
        const uint8_t* xb,
        idx_t k,
        const idx_t* assign,
        const int32_t* centroid_dis,
        int32_t* distances,
        idx_t* labels,
        int store_pairs,
        const FaissSearchParametersIVF* params);

int faiss_IndexBinaryIVF_get_centroids_and_cardinality(
        const FaissIndexBinaryIVF* index,
        uint8_t* centroid_vectors,
        size_t* cardinalities,
        idx_t* centroid_ids);

#ifdef __cplusplus
}
#endif

#endif
