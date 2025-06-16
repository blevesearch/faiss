extern "C" {

int faiss_IndexBinaryIVF_set_direct_map(
        FaissIndexBinaryIVF* index,
        int direct_map_type) {
    try {
        reinterpret_cast<faiss::IndexBinaryIVF*>(index)->set_direct_map_type(
                static_cast<faiss::DirectMap::Type>(direct_map_type));
    }
    CATCH_AND_HANDLE
}

int faiss_get_lists_for_keys_binary(
        FaissIndexBinaryIVF* index,
        idx_t* keys,
        size_t n_keys,
        idx_t* lists) {
    try {
        reinterpret_cast<faiss::IndexBinaryIVF*>(index)->get_lists_for_keys(
                keys, n_keys, lists);
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
        faiss::IndexBinaryIVF* index_bivf = reinterpret_cast<IndexBinaryIVF*>(index);
        assert(index_bivf);

        index_bivf->quantizer->search(
                n,
                query,
                k,
                centroid_distances,
                centroid_ids,
                reinterpret_cast<const faiss::SearchParameters*>(params));
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
        faiss::IndexBinaryIVF* index_bivf = reinterpret_cast<IndexBinaryIVF*>(index);
        assert(index_bivf);

        index_bivf->search_preassigned(n, x, k, assign, centroid_dis, distances, 
        labels, store_pairs, reinterpret_cast<const faiss::SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}



}