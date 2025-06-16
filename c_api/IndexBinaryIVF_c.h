#ifndef FAISS_INDEX_BINARY_IVF_C_H
#define FAISS_INDEX_BINARY_IVF_C_H

#include "Index_c.h"
#include "faiss_c.h"
#include "macros_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

int faiss_IndexBinaryIVF_set_direct_map(
        FaissIndexBinaryIVF* index,
        int direct_map_type);

int faiss_get_lists_for_keys_binary(
        FaissIndexBinaryIVF* index,
        idx_t* keys,
        size_t n_keys,
        idx_t* lists) ; 


int faiss_Search_closest_eligible_centroids_binary(
        FaissIndexBinaryIVF* index,
        idx_t n,
        const uint8_t* query,
        idx_t k,
        int32_t* centroid_distances,
        idx_t* centroid_ids,
        const FaissSearchParameters* params);

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
        const FaissSearchParametersIVF* params);

DEFINE_GETTER_PERMISSIVE(IndexBinaryIVF, FaissIndexBinary*, quantizer);
#ifdef __cplusplus
}
#endif

#endif