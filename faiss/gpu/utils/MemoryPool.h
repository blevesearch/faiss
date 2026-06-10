/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>

namespace faiss {
namespace gpu {

/// A memory pool scoped to a single GPU device with a soft capacity limit.
/// Supports stream-ordered allocation and deallocation of device memory.
class GpuMemoryPool {
   public:
    /// Allocate a new region of memory that we manage
    GpuMemoryPool(int device, size_t poolCap);

    ~GpuMemoryPool();

    int getDevice() const;

    /// All allocations requested should be a multiple of 16 bytes
    void* allocMemory(cudaStream_t stream, size_t size);
    void deallocMemory(int device, cudaStream_t, size_t size, void* p);

    size_t getSizeAvailable() const;

   protected:
    /// Our device
    int device_;

    /// Soft limit for the pool size
    size_t poolCap_;

    /// Our pool
    cudaMemPool_t pool_;
};

} // namespace gpu
} // namespace faiss
