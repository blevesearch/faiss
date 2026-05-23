/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_GPU_INDEX_IVF_C_EX_H
#define FAISS_GPU_INDEX_IVF_C_EX_H

#include <stddef.h>

#include "GpuIndex_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS_INHERITED(GpuIndexIVF, GpuIndex)

/// Getter for nlist (number of inverted lists)
FAISS_DECLARE_GETTER(GpuIndexIVF, size_t, nlist)

/** Reserve memory on the GPU for adding vectors to an IVF index.
 *
 * This function allows the caller to pre-reserve GPU memory before
 * adding vectors to the index, which can curb the geometric growth
 * pattern of GPU memory usage due to appending vectors to the inverted lists.
 * 
 * @param index       opaque pointer to GPU index object
 * @param n           number of inverted lists (nlist)
 * @param x           number of vectors assigned per list, size n
 */
int faiss_GpuIndexIVF_reserve_assigned_memory(
        FaissGpuIndexIVF* index,
        size_t n,
        const idx_t* x);

/** Compute the amount of memory required for a given set of assignments.
 *
 * This function allows the caller to compute the amount of GPU memory 
 * that would be required by an IVF index given the number of inverted 
 * lists and the number of vectors assigned per list.
 * 
 * @param index       opaque pointer to GPU index object
 * @param n           number of inverted lists (nlist)
 * @param x           number of vectors assigned per list, size n
 * @param out         output parameter for the amount of memory required in bytes
 */
int faiss_GpuIndexIVF_compute_required_memory(
        FaissGpuIndexIVF* index,
        size_t n,
        const idx_t* x,
        size_t* out);
 
/** Return the centroid assignments for the input vectors.
 *
 * This function returns the inverted list assignments for
 * the input vectors, and must be called after the IVF index is trained
 * 
 * @param index       opaque pointer to GPU index object
 * @param n           number of vectors
 * @param x           input vectors to search, size n * d
 * @param labels      output labels of the NNs, size n
 */
int faiss_GpuIndexIVF_assign(
        const FaissGpuIndexIVF* index, 
        idx_t n, 
        const float* x, 
        idx_t* labels);

/// Downcast a GpuIndex to GpuIndexIVF (returns NULL if not a GpuIndexIVF)
FaissGpuIndexIVF* faiss_GpuIndexIVF_cast(FaissGpuIndex* index);

#ifdef __cplusplus
}
#endif

#endif
