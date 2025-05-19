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
#include "faiss_c.h"
#include "Index_c.h"


#ifdef __cplusplus
extern "C" {
#endif

int faiss_Index_reconstruct_batch(const FaissIndex* index, idx_t n,
        const idx_t* keys, float* recons);


int faiss_Index_merge_from(FaissIndex* index, FaissIndex* other, idx_t add_id);

size_t faiss_Index_size(FaissIndex* index);

#ifdef __cplusplus
}
#endif

#endif