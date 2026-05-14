/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "GpuIndexIVF_c.h"
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include "macros_impl.h"

using faiss::gpu::GpuIndexIVF;
using faiss::gpu::GpuIndexIVFFlat;
using faiss::gpu::GpuIndexIVFPQ;
using faiss::gpu::GpuIndexIVFScalarQuantizer;

int faiss_GpuIndexIVF_reserve_memory(
        FaissGpuIndexIVF* index,
        size_t num_vectors) {
    try {
        auto ivf = reinterpret_cast<GpuIndexIVF*>(index);
        if (auto flat = dynamic_cast<GpuIndexIVFFlat*>(ivf)) {
            flat->reserveMemory(num_vectors);
        } else if (auto pq = dynamic_cast<GpuIndexIVFPQ*>(ivf)) {
            pq->reserveMemory(num_vectors);
        } else if (auto sq = dynamic_cast<GpuIndexIVFScalarQuantizer*>(ivf)) {
            sq->reserveMemory(num_vectors);
        } else {
            throw std::runtime_error(
                    "reserve_memory not supported for this GpuIndexIVF subtype");
        }
    }
    CATCH_AND_HANDLE
}
