/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2023-present Facebook. All Rights Reserved.
// -*- c -*-

#ifndef FAISS_INDEX_IVF_EX_C_H
#define FAISS_INDEX_IVF_EX_C_H

#include "Clustering_c.h"
#include "Index_c.h"
#include "faiss_c.h"
#include "IndexIVF_c.h"

#ifdef __cplusplus
extern "C" {
#endif
int faiss_IndexIVF_set_direct_map(
        FaissIndexIVF* index,
        int direct_map_type);

int faiss_SearchParametersIVF_new_with_sel(
        FaissSearchParametersIVF** p_sp,
        FaissIDSelector* sel);

#ifdef __cplusplus
}
#endif

#endif