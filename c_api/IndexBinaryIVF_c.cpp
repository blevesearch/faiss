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
}