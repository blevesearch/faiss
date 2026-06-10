/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#include "GpuMemoryPool_c.h"

using faiss::gpu::GpuMemoryPool;

int faiss_GpuMemoryPool_new(
        int device,
        size_t poolCap,
        FaissGpuMemoryPool** out) {
    try {
        auto pool = new GpuMemoryPool(device, poolCap);
        *out = reinterpret_cast<FaissGpuMemoryPool*>(pool);
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(GpuMemoryPool)
