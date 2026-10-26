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

// faiss_IndexBinary_search_with_params is declared in IndexBinary_c.h

/** Return an approximate size estimate in bytes for the index.
 * 
 *
 * @param index     opaque pointer to index object
 * @param p_size    output approximate size in bytes
 */
int faiss_IndexBinary_size(const FaissIndexBinary* index, size_t* p_size);

/** The size of the produced codes in bytes.
 *
 * @param index   opaque pointer to index object
 * @param size    the returned size in bytes
 */
int faiss_IndexBinary_sa_code_size(const FaissIndexBinary* index, size_t* size);

#ifdef __cplusplus
}
#endif

#endif
