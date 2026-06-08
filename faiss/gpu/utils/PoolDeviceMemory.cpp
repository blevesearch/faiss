/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/PoolDeviceMemory.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu {

PoolDeviceMemory::PoolDeviceMemory(int device) : device_(device) {
    CUDA_VERIFY(cudaDeviceGetDefaultMemPool(&pool_, device_));
}

PoolDeviceMemory::~PoolDeviceMemory() {}

int PoolDeviceMemory::getDevice() const {
    return device_;
}

void* PoolDeviceMemory::allocMemory(cudaStream_t stream, size_t size) {
    void* ptr = nullptr;
    auto err = cudaMallocAsync(&ptr, size, stream);
    if (err != cudaSuccess) {
        cudaGetLastError();
        FAISS_THROW_IF_NOT_FMT(
                err == cudaSuccess, "%s", cudaGetErrorString(err));
    }
    return ptr;
}

void PoolDeviceMemory::deallocMemory(
        int device,
        cudaStream_t stream,
        size_t size,
        void* p) {
    if (p) {
        FAISS_ASSERT(device == device_);
        CUDA_VERIFY(cudaFreeAsync(p, stream));
    }
}

size_t PoolDeviceMemory::getSizeAvailable() const {
    size_t reserved = 0, used = 0;
    CUDA_VERIFY(cudaMemPoolGetAttribute(
            pool_, cudaMemPoolAttrReservedMemCurrent, &reserved));
    CUDA_VERIFY(cudaMemPoolGetAttribute(
            pool_, cudaMemPoolAttrUsedMemCurrent, &used));
    return reserved > used ? reserved - used : 0;
}

} // namespace gpu
} // namespace faiss
