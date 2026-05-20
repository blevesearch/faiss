/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_GPU_INDEX_C_H
#define FAISS_GPU_INDEX_C_H

#include "../faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS(GpuIndexConfig)

FAISS_DECLARE_GETTER(GpuIndexConfig, int, device)

FAISS_DECLARE_CLASS_INHERITED(GpuIndex, Index)

/// Getter for d (dimension)
FAISS_DECLARE_GETTER(GpuIndex, int, d)

/** Perform training on a representative set of vectors
 *
 * @param index  opaque pointer to GPU index object
 * @param n      nb of training vectors
 * @param x      training vectors, size n * d
 */
int faiss_GpuIndex_train(FaissGpuIndex* index, idx_t n, const float* x);

/** Add n vectors of dimension d to the index.
 *
 * @param index  opaque pointer to GPU index object
 * @param n      nb of vectors to add
 * @param x      input vectors, size n * d
 */
int faiss_GpuIndex_add(FaissGpuIndex* index, idx_t n, const float* x);

/** Query n vectors of dimension d to the index.
 *
 * @param index       opaque pointer to GPU index object
 * @param n           number of vectors to search
 * @param x           input vectors to search, size n * d
 * @param k           number of nearest neighbors to return
 * @param distances   output pairwise distances, size n * k
 * @param labels      output labels of the NNs, size n * k
 */
int faiss_GpuIndex_search(
        const FaissGpuIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels);

/** Return an approximate size estimate in bytes for the index.
 * 
 *
 * @param index     opaque pointer to index object
 * @param p_size    output approximate size in bytes
 */
int faiss_GpuIndex_size(const FaissGpuIndex* index, size_t* p_size);

/// Free the GPU index
FAISS_DECLARE_DESTRUCTOR(GpuIndex)

#ifdef __cplusplus
}
#endif

#endif
