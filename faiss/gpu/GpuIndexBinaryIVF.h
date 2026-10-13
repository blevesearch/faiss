/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Clustering.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/GpuResources.h>
#include <memory>
#include <vector>

namespace faiss {
namespace gpu {

class BinaryIVF;
class GpuIndexBinaryFlat;

struct GpuIndexBinaryIVFConfig : public GpuIndexConfig {
    /// Index storage options for the GPU. INDICES_IVF does not retain the
    /// user ids, so such an index cannot be copied back to the CPU.
    IndicesOptions indicesOptions = INDICES_64_BIT;
};

/// A GPU version of IndexBinaryIVF: an inverted file index over bit vectors
/// searched via Hamming distance. The coarse quantizer is a
/// GpuIndexBinaryFlat that is owned by this index.
///
/// Unlike the CPU index, a direct map (and thus reconstruct()) and
/// remove_ids() are not supported.
class GpuIndexBinaryIVF : public IndexBinary {
   public:
    /// Construct from a pre-existing faiss::IndexBinaryIVF instance, copying
    /// data over to the given GPU
    GpuIndexBinaryIVF(
            GpuResourcesProvider* provider,
            const faiss::IndexBinaryIVF* index,
            GpuIndexBinaryIVFConfig config = GpuIndexBinaryIVFConfig());

    /// Construct an empty, untrained instance with `nlist` inverted lists
    /// over `dims`-bit vectors
    GpuIndexBinaryIVF(
            GpuResourcesProvider* provider,
            int dims,
            idx_t nlist,
            GpuIndexBinaryIVFConfig config = GpuIndexBinaryIVFConfig());

    ~GpuIndexBinaryIVF() override;

    /// Returns the device that this index is resident on
    int getDevice() const;

    /// Returns the coarse quantizer (owned by this index), e.g. to inspect
    /// the centroids
    GpuIndexBinaryFlat* getQuantizer();

    /// Sets the smallest amount of host data (in bytes) for which add and
    /// search are performed in pages, to bound the device memory used
    void setMinPagingSize(size_t size);

    /// Returns the current minimum paging size
    size_t getMinPagingSize() const;

    /// Returns a reference to our GpuResources object that manages memory,
    /// stream and handle resources on the GPU
    std::shared_ptr<GpuResources> getResources();

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss::IndexBinaryIVF* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance. The coarse quantizer is replaced by an
    /// IndexBinaryFlat owned by `index`.
    void copyTo(faiss::IndexBinaryIVF* index) const;

    /// Trains the coarse quantizer: binary k-means whose assignment step runs
    /// on the GPU, controlled by `cp`
    void train(idx_t n, const uint8_t* x) override;

    /// Adds `n` vectors; `x` may be on the host or the device
    void add(idx_t n, const uint8_t* x) override;

    /// Adds `n` vectors with the given ids (host or device resident)
    void add_with_ids(idx_t n, const uint8_t* x, const idx_t* xids) override;

    void reset() override;

    /// `params` may be null or an IVFSearchParameters (to override nprobe)
    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Returns the number of inverted lists we're managing
    idx_t getNumLists() const;

    /// Returns the number of vectors present in a particular inverted list
    idx_t getListLength(idx_t listId) const;

    /// Return the codes contained in a particular inverted list, for
    /// debugging purposes
    std::vector<uint8_t> getListVectorData(idx_t listId) const;

    /// Return the vector ids contained in a particular inverted list, for
    /// debugging purposes
    std::vector<idx_t> getListIndices(idx_t listId) const;

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(size_t numVecs);

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory();

   public:
    /// Number of inverted lists
    idx_t nlist;

    /// Number of inverted lists probed at query time
    size_t nprobe = 1;

    /// Clustering parameters used by train()
    ClusteringParameters cp;

   protected:
    /// (Re)creates the empty GPU inverted lists
    void resetInvertedLists_();

    /// Adds one page of vectors (host resident codes and ids)
    void addPage_(idx_t n, const uint8_t* x, const idx_t* xids);

    /// Searches one page of queries
    void searchPage_(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            size_t nprobe,
            int32_t* distances,
            idx_t* labels) const;

    /// Manages streams, cuBLAS handles and scratch memory for devices
    std::shared_ptr<GpuResources> resources_;

    /// Configuration options
    const GpuIndexBinaryIVFConfig ivfConfig_;

    /// Size in bytes above which host data is paged to the device
    size_t minPagedSize_;

    /// Coarse quantizer holding the nlist centroids
    std::unique_ptr<GpuIndexBinaryFlat> quantizer_;

    /// The on-device inverted lists
    std::unique_ptr<BinaryIVF> index_;
};

} // namespace gpu
} // namespace faiss
