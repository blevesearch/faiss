/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

/// Scans the inverted lists selected in `listIds` for each query, computing
/// Hamming distances between the binary query codes and the list codes, and
/// k-selects the smallest distances. Lists are stored non-interleaved
/// (vector-major), with `queries.getSize(1)` bytes per code.
/// Entries that could not be filled (fewer than k candidates) are returned
/// as distance INT32_MAX and index -1.
void runIVFBinaryScan(
        Tensor<uint8_t, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
        int k,
        // output
        Tensor<int32_t, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

} // namespace gpu
} // namespace faiss
