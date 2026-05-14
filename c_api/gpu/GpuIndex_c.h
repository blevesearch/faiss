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

#ifdef __cplusplus
}
#endif

#endif
