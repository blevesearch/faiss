/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_GPU_MEMORY_POOL_C_H
#define FAISS_GPU_MEMORY_POOL_C_H

#include <cuda_runtime_api.h>
#include <faiss/gpu/utils/MemoryPool.h>
#include "GpuIndex_c.h"
#include "macros_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS(GpuMemoryPool)

/**
 * Returns a memory pool for the given device and pool capacity. 
 * The pool will be used for temporary memory allocations on the GPU, and will automatically release memory back to the system when the pool capacity is reached. 
 * The pool may exceed the specified capacity if necessary, but will not release memory until 
 * the capacity is reached.
 * 
 * @param device The GPU device ID.
 * @param poolCap The maximum capacity of the memory pool in bytes.
 * @param out A pointer to the memory pool object.
 * @return 0 on success, non-zero on failure.
*/
int faiss_GpuMemoryPool_new(
		int device,
		size_t poolCap,
		FaissGpuMemoryPool** out);

FAISS_DECLARE_DESTRUCTOR(GpuMemoryPool)

#ifdef __cplusplus
}
#endif
#endif