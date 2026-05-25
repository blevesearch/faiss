/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "GpuIndex_c_ex.h"
#include <faiss/gpu/GpuIndex.h>
#include "macros_impl.h"

using faiss::gpu::GpuIndex;

DEFINE_GETTER(GpuIndex, int, d)

int faiss_GpuIndex_train(FaissGpuIndex* index, idx_t n, const float* x) {
    try {
        reinterpret_cast<GpuIndex*>(index)->train(n, x);
    }
    CATCH_AND_HANDLE
}

int faiss_GpuIndex_add(FaissGpuIndex* index, idx_t n, const float* x) {
    try {
        reinterpret_cast<GpuIndex*>(index)->add(n, x);
    }
    CATCH_AND_HANDLE
}

int faiss_GpuIndex_search(
        const FaissGpuIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) {
    try {
        reinterpret_cast<const GpuIndex*>(index)->search(
                n, x, k, distances, labels);
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(GpuIndex)
