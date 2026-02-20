/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c -*-

#ifndef FAISS_INDEX_BINARY_C_EX_H
#define FAISS_INDEX_BINARY_C_EX_H

#include "IndexBinary_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** query n vectors of dimension d to the index.
 *
 * return at most k vectors. If there are not enough results for a
 * query, the result array is padded with -1s.
 *
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d
 * @param k           number of results to return
 * @param params      search parameters
 * @param distances   output distances, size n*k
 * @param labels      output labels of the NNs, size n*k
 */
int faiss_IndexBinary_search_with_params(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        const FaissSearchParameters* params,
        int32_t* distances,
        idx_t* labels);

#ifdef __cplusplus
}
#endif

#endif
