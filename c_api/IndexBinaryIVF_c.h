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

DEFINE_GETTER_PERMISSIVE(IndexBinaryIVF, FaissIndexBinary*, quantizer);
#ifdef __cplusplus
}
#endif

#endif