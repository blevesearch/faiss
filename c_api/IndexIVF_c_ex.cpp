/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2023-present Facebook. All Rights Reserved.
// -*- c++ -*-
#include "IndexIVF_c_ex.h"
#include <faiss/IndexIVF.h>
#include "macros_impl.h"

using faiss::IndexIVF;
using faiss::SearchParametersIVF;

int faiss_IndexIVF_set_direct_map(FaissIndexIVF* index, int direct_map_type) {
    try {
        reinterpret_cast<IndexIVF*>(index)->set_direct_map_type(
                static_cast<faiss::DirectMap::Type>(direct_map_type));
    }
    CATCH_AND_HANDLE
}

int faiss_SearchParametersIVF_new_with_sel(
        FaissSearchParametersIVF** p_sp,
        FaissIDSelector* sel) {
    try {
        SearchParametersIVF* sp = new SearchParametersIVF;
        sp->sel = reinterpret_cast<faiss::IDSelector*>(sel);
        *p_sp = reinterpret_cast<FaissSearchParametersIVF*>(sp);
    }
    CATCH_AND_HANDLE
}