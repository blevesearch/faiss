/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c -*-

#include "IndexBinary_c_ex.h"
#include <faiss/IndexBinary.h>
#include "macros_impl.h"

extern "C" {
int faiss_IndexBinary_search_with_params(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        const FaissSearchParameters* params,
        int32_t* distances,
        idx_t* labels) {
    try {
        reinterpret_cast<const faiss::IndexBinary*>(index)->search(
                n,
                x,
                k,
                distances,
                labels,
                reinterpret_cast<const faiss::SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}

size_t faiss_IndexBinary_size(FaissIndexBinary* index) {
    auto xIndex = reinterpret_cast<faiss::IndexBinary*>(index);
    size_t rv = sizeof(xIndex);
    return rv;
}
}
