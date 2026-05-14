/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_GPU_INDEX_IVF_C_H
#define FAISS_GPU_INDEX_IVF_C_H

#include <stddef.h>

#include "GpuIndex_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS_INHERITED(GpuIndexIVF, GpuIndex)

int faiss_GpuIndexIVF_reserve_memory(
        FaissGpuIndexIVF* index,
        size_t num_vectors);

#ifdef __cplusplus
}
#endif

#endif
