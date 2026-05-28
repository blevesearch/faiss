/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "GpuIndexIVF_c_ex.h"
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include "macros_impl.h"

using faiss::gpu::GpuIndex;
using faiss::gpu::GpuIndexIVF;
using faiss::gpu::GpuIndexIVFFlat;
using faiss::gpu::GpuIndexIVFPQ;
using faiss::gpu::GpuIndexIVFScalarQuantizer;

DEFINE_GETTER(GpuIndexIVF, size_t, nlist)

int faiss_GpuIndexIVF_reserve_assigned_memory(
        FaissGpuIndexIVF* index,
        size_t n,
        const idx_t* x) {
    try {
        auto ivf = reinterpret_cast<GpuIndexIVF*>(index);
        if (auto sq = dynamic_cast<GpuIndexIVFScalarQuantizer*>(ivf)) {
            sq->reserveAssignedMemory(n, x);
        } else if (auto pq = dynamic_cast<GpuIndexIVFPQ*>(ivf)) {
            pq->reserveAssignedMemory(n, x);
        } else if (auto flat = dynamic_cast<GpuIndexIVFFlat*>(ivf)) {
            flat->reserveAssignedMemory(n, x);
        }
    }
    CATCH_AND_HANDLE
}

int faiss_GpuIndexIVF_compute_required_memory(
        FaissGpuIndexIVF* index,
        size_t n,
        const idx_t* x,
        size_t* out) {
    try {
        auto ivf = reinterpret_cast<GpuIndexIVF*>(index);
        if (auto sq = dynamic_cast<GpuIndexIVFScalarQuantizer*>(ivf)) {
            sq->computeRequiredMemory(n, x, out);
        } else if (auto pq = dynamic_cast<GpuIndexIVFPQ*>(ivf)) {
            pq->computeRequiredMemory(n, x, out);
        } else if (auto flat = dynamic_cast<GpuIndexIVFFlat*>(ivf)) {
            flat->computeRequiredMemory(n, x, out);
        }
    }
    CATCH_AND_HANDLE
}

int faiss_GpuIndexIVF_assign(
        const FaissGpuIndexIVF* index,
        idx_t n,
        const float* x,
        idx_t* labels) {
    try {
        reinterpret_cast<const GpuIndexIVF*>(index)->quantizer->assign(
                n, x, labels);
    }
    CATCH_AND_HANDLE
}

FaissGpuIndexIVF* faiss_GpuIndexIVF_cast(FaissGpuIndex* index) {
    return reinterpret_cast<FaissGpuIndexIVF*>(
            dynamic_cast<GpuIndexIVF*>(reinterpret_cast<GpuIndex*>(index)));
}
