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

/// Device memory manager that provides temporary memory allocations
/// out of a region of memory, for a single device
class PoolDeviceMemory {
   public:
    /// Allocate a new region of memory that we manage
    PoolDeviceMemory(int device);

    ~PoolDeviceMemory();

    int getDevice() const;

    /// All allocations requested should be a multiple of 16 bytes
    void* allocMemory(cudaStream_t stream, size_t size);
    void deallocMemory(int device, cudaStream_t, size_t size, void* p);

    size_t getSizeAvailable() const;

   protected:
    /// Our device
    int device_;

    /// Our pool
    cudaMemPool_t pool_;
};

} // namespace gpu
} // namespace faiss
