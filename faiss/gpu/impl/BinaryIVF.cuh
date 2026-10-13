/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/impl/IVFBase.cuh>

namespace faiss {
namespace gpu {

/// Inverted lists of binary codes (Hamming distance) resident on the GPU.
/// Codes are stored non-interleaved, `dim / 8` bytes per vector. All
/// IndicesOptions are supported for the user indices.
class BinaryIVF : public IVFBase {
   public:
    /// `dim` is the number of bits per vector, and must be a multiple of 8
    BinaryIVF(
            GpuResources* resources,
            int dim,
            idx_t nlist,
            IndicesOptions indicesOptions,
            MemorySpace space);

    ~BinaryIVF() override;

    /// Number of bytes in a single code
    size_t getCodeSize() const;

    /// Appends `n` codes with the given ids to the given inverted lists.
    /// All inputs are resident on the host. Entries whose list number is
    /// negative are skipped. Returns the number of codes added.
    idx_t addCodes(
            const uint8_t* codes,
            const idx_t* ids,
            const idx_t* listNos,
            idx_t n);

    /// Scans the `nprobe` lists per query in `listIds` (chosen by the coarse
    /// quantizer) and returns the k nearest codes by Hamming distance.
    /// All tensors are resident on the device.
    void searchBinary(
            Tensor<uint8_t, 2, true>& queries,
            Tensor<idx_t, 2, true>& listIds,
            int k,
            Tensor<int32_t, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices);

    // The float-based IVFBase entry points are not meaningful for binary
    // codes
    void search(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& queries,
            int nprobe,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices) override;

    void searchPreassigned(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& vecs,
            Tensor<float, 2, true>& ivfDistances,
            Tensor<idx_t, 2, true>& ivfAssignments,
            int k,
            Tensor<float, 2, true>& outDistances,
            Tensor<idx_t, 2, true>& outIndices,
            bool storePairs) override;

    idx_t addVectors(
            Index* coarseQuantizer,
            Tensor<float, 2, true>& vecs,
            Tensor<idx_t, 1, true>& indices) override;

    void updateQuantizer(Index* quantizer) override;

    void reconstruct_n(idx_t i0, idx_t n, float* out) override;

   protected:
    size_t getGpuVectorsEncodingSize_(idx_t numVecs) const override;
    size_t getCpuVectorsEncodingSize_(idx_t numVecs) const override;

    /// The GPU and CPU layouts are identical
    std::vector<uint8_t> translateCodesToGpu_(
            std::vector<uint8_t> codes,
            idx_t numVecs) const override;
    std::vector<uint8_t> translateCodesFromGpu_(
            std::vector<uint8_t> codes,
            idx_t numVecs) const override;

    void appendVectors_(
            Tensor<float, 2, true>& vecs,
            Tensor<float, 2, true>& ivfCentroidResiduals,
            Tensor<idx_t, 1, true>& indices,
            Tensor<idx_t, 1, true>& uniqueLists,
            Tensor<idx_t, 1, true>& vectorsByUniqueList,
            Tensor<idx_t, 1, true>& uniqueListVectorStart,
            Tensor<idx_t, 1, true>& uniqueListStartOffset,
            Tensor<idx_t, 1, true>& listIds,
            Tensor<idx_t, 1, true>& listOffset,
            cudaStream_t stream) override;
};

} // namespace gpu
} // namespace faiss
