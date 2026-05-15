/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c -*-

#ifndef FAISS_INDEX_EX_C_H
#define FAISS_INDEX_EX_C_H

#include <stddef.h>
#include <stdio.h>
#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

int faiss_Index_reconstruct_batch(
        const FaissIndex* index,
        idx_t n,
        const idx_t* keys,
        float* recons);

int faiss_Index_merge_from(FaissIndex* index, FaissIndex* other, idx_t add_id);

/** Return an approximate size estimate in bytes for the index.
 * 
 * The estimate accounts for stored codes, the base struct size, and for
 * IVF-based indices, includes centroid and stored ID overhead.
 *
 * @param index     opaque pointer to index object
 * @param p_size    output approximate size in bytes
 */
int faiss_Index_size(const FaissIndex* index, size_t* p_size);

/** Return the static struct size of the index in bytes.
 *
 * This returns only the base struct footprint 
 * without accounting for any stored data
 *
 * @param index     opaque pointer to index object
 * @param p_size    output static size in bytes
 */
int faiss_Index_static_size(const FaissIndex* index, size_t* p_size);

/** Compute distances between a query vector and a set of vectors 
 *
 * @param index       opaque pointer to index object
 * @param query       query vector, size d
 * @param ids         array of vector ids to compute distances to
 * @param n_ids       number of ids in the array
 * @param distances   output distances, size n_ids
 */
int faiss_Index_dist_compute(
        const FaissIndex* index,
        const float* query,
        const idx_t* ids,
        size_t n_ids,
        float* distances);

#ifdef __cplusplus
}
#endif

#endif