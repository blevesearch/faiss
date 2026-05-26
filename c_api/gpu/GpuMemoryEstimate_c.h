/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_GPU_MEMORY_ESTIMATE_C_H
#define FAISS_GPU_MEMORY_ESTIMATE_C_H

#include <stddef.h>

#include "../Index_c.h"
#include "../faiss_c.h"
#include "GpuClonerOptions_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Estimate the GPU memory required to clone a CPU faiss index to a single
 * GPU device, without actually performing the clone.
 *
 * This computes the size of the device-side index payload, accounting for:
 *   - the interleaved layout used by GpuIndexIVFFlat / GpuIndexIVFScalarQuantizer
 *     (block size = warp_size of the target GPU)
 *   - the flat encoding used by the default GpuIndexIVFPQ build
 *   - the coarse quantizer (centroids) storage (float32 or float16)
 *   - per-list index storage (INDICES_32_BIT / INDICES_64_BIT)
 *   - optional precomputed tables for IVFPQ (when options->usePrecomputed)
 *   - the flat data when the input is a top-level IndexFlat
 *
 * Supports IndexFlat, IndexIVFFlat, IndexIVFScalarQuantizer, IndexIVFPQ and
 * transparently unwraps IndexIDMap / IndexIDMap2 / IndexPreTransform.
 *
 * @param index   the CPU index that would be cloned (must not be NULL)
 * @param device  the target GPU device (used to read its warp size)
 * @param options the cloner options that would be passed to the clone call.
 *                If NULL, faiss::gpu::GpuClonerOptions defaults are used.
 * @param out     output: estimated bytes required on the target GPU. Set to 0
 *                if the index type cannot be cloned to GPU; in that case the
 *                call still returns 0 (success).
 */
int faiss_GpuMemoryEstimate_for_cpu_index(
        const FaissIndex* index,
        int device,
        const FaissGpuClonerOptions* options,
        size_t* out);

#ifdef __cplusplus
}
#endif

#endif
