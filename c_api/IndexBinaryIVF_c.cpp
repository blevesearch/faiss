#include "IndexBinaryIVF_c.h"
#include <faiss/IndexBinaryIVF.h>
#include "macros_impl.h"

using faiss::IndexBinaryIVF;

DEFINE_GETTER_PERMISSIVE(IndexBinaryIVF, FaissIndexBinaryPtr, quantizer)

extern "C" {

int faiss_IndexBinaryIVF_set_direct_map(
        FaissIndexBinaryIVF* index,
        int direct_map_type) {
    try {
        reinterpret_cast<IndexBinaryIVF*>(index)->set_direct_map_type(
                static_cast<faiss::DirectMap::Type>(direct_map_type));
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_get_lists_for_keys_binary(
        FaissIndexBinaryIVF* index,
        idx_t* keys,
        size_t n_keys,
        idx_t* lists) {
    try {
        reinterpret_cast<IndexBinaryIVF*>(index)->get_lists_for_keys(
                keys, n_keys, lists);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_Search_closest_eligible_centroids_binary(
        FaissIndexBinaryIVF* index,
        idx_t n,
        const uint8_t* query,
        idx_t k,
        int32_t* centroid_distances,
        idx_t* centroid_ids,
        const FaissSearchParameters* params) {
    try {
        auto idx = reinterpret_cast<IndexBinaryIVF*>(index);
        idx->quantizer->search(
                n,
                query,
                k,
                centroid_distances,
                centroid_ids,
                reinterpret_cast<const faiss::SearchParameters*>(params));

        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinaryIVF_set_is_trained(FaissIndexBinaryIVF* index, int is_trained) {
    try {
        reinterpret_cast<faiss::IndexBinaryIVF*>(index)->is_trained = static_cast<bool>(is_trained);
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
        auto idx = reinterpret_cast<const IndexBinaryIVF*>(index);
        idx->search_preassigned(
                n, x, k, assign, centroid_dis, distances, labels,
                static_cast<bool>(store_pairs),
                reinterpret_cast<const faiss::IVFSearchParameters*>(params));
        return 0;
    }
    CATCH_AND_HANDLE
}

} // extern "C"