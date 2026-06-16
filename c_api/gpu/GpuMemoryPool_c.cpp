/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#include "GpuMemoryPool_c.h"
#include <faiss/gpu/utils/MemoryPool.h>
#include "macros_impl.h"

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

void faiss_GpuMemoryPool_free(FaissGpuMemoryPool* obj) {
    delete reinterpret_cast<GpuMemoryPool*>(obj);
}
