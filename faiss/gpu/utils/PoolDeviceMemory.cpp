/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/PoolDeviceMemory.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu {

PoolDeviceMemory::PoolDeviceMemory(
        GpuResources* res,
        int device,
        MemorySpace space)
        : device_(device) {
    FAISS_ASSERT(space == MemorySpace::Device);

    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device_;

    CUDA_VERIFY(cudaMemPoolCreate(&pool_, &props));
}

PoolDeviceMemory::~PoolDeviceMemory() {
    cudaMemPoolDestroy(pool_);
}

int PoolDeviceMemory::getDevice() const {
    return device_;
}

void* PoolDeviceMemory::allocMemory(cudaStream_t stream, size_t size) {
    void* ptr = nullptr;
    auto err = cudaMemPoolAllocAsync(&ptr, pool_, size, stream);
	if (err != cudaSuccess) {
		cudaGetLastError();
		FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "%s", str.c_str());
	}
    return ptr;
}

void PoolDeviceMemory::deallocMemory(
        int device,
        cudaStream_t stream,
        size_t size,
        void* p) {
    FAISS_ASSERT(device == device_);
    if (p) {
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