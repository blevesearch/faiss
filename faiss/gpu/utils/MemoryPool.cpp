/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/MemoryPool.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu {

GpuMemoryPool::GpuMemoryPool(int device, size_t poolCap)
        : device_(device), poolCap_(poolCap) {
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = device;
    poolProps.location.type = cudaMemLocationTypeDevice;
    auto err = cudaMemPoolCreate(&pool_, &poolProps);
    if (err != cudaSuccess) {
        FAISS_THROW_IF_NOT_FMT(
                err == cudaSuccess, "%s", cudaGetErrorString(err));
    }
    // poolCap_ is our "soft" limit for the pool size, we start releasing memory
    // back to the system when we reach that threshold, but we can exceed it if
    // necessary
    err = cudaMemPoolSetAttribute(
            pool_, cudaMemPoolAttrReleaseThreshold, &poolCap_);
    if (err != cudaSuccess) {
        FAISS_THROW_IF_NOT_FMT(
                err == cudaSuccess, "%s", cudaGetErrorString(err));
    }
}

GpuMemoryPool::~GpuMemoryPool() {
    if (pool_) {
        CUDA_VERIFY(cudaMemPoolDestroy(pool_));
    }
}

int GpuMemoryPool::getDevice() const {
    return device_;
}

void* GpuMemoryPool::allocMemory(cudaStream_t stream, size_t size) {
    void* ptr = nullptr;
    auto err = cudaMallocFromPoolAsync(&ptr, size, pool_, stream);
    if (err != cudaSuccess) {
        cudaGetLastError();
        FAISS_THROW_IF_NOT_FMT(
                err == cudaSuccess, "%s", cudaGetErrorString(err));
    }
    return ptr;
}

void GpuMemoryPool::deallocMemory(
        int device,
        cudaStream_t stream,
        size_t size,
        void* p) {
    if (p) {
        FAISS_ASSERT(device == device_);
        CUDA_VERIFY(cudaFreeAsync(p, stream));
    }
}

size_t GpuMemoryPool::getSizeAvailable() const {
    size_t used = 0;
    CUDA_VERIFY(cudaMemPoolGetAttribute(
            pool_, cudaMemPoolAttrUsedMemCurrent, &used));
    return poolCap_ > used ? poolCap_ - used : 0;
}

} // namespace gpu
} // namespace faiss
