/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndexBinaryIVF.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/gpu/GpuIndexBinaryFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/gpu/impl/BinaryIVF.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>

#include <algorithm>
#include <numeric>

namespace faiss {
namespace gpu {

namespace {

/// Default size in bytes above which host data is paged to the device
constexpr size_t kMinPageSize = (size_t)256 * 1024 * 1024;

/// Creates the (empty) coarse quantizer, which always uses the native GPU
/// path, sharing our resources
std::unique_ptr<GpuIndexBinaryFlat> makeQuantizer(
        const std::shared_ptr<GpuResources>& resources,
        int dims,
        const GpuIndexBinaryIVFConfig& config) {
    GpuIndexBinaryFlatConfig flatConfig;
    flatConfig.device = config.device;
    flatConfig.memorySpace = config.memorySpace;
    flatConfig.use_cuvs = false;

    GpuResourcesProviderFromInstance provider(resources);
    return std::unique_ptr<GpuIndexBinaryFlat>(
            new GpuIndexBinaryFlat(&provider, dims, flatConfig));
}

} // namespace

GpuIndexBinaryIVF::GpuIndexBinaryIVF(
        GpuResourcesProvider* provider,
        const faiss::IndexBinaryIVF* index,
        GpuIndexBinaryIVFConfig config)
        : IndexBinary(index->d),
          nlist(index->nlist),
          resources_(provider->getResources()),
          ivfConfig_(config),
          minPagedSize_(kMinPageSize) {
    copyFrom(index);
}

GpuIndexBinaryIVF::GpuIndexBinaryIVF(
        GpuResourcesProvider* provider,
        int dims,
        idx_t nlist,
        GpuIndexBinaryIVFConfig config)
        : IndexBinary(dims),
          nlist(nlist),
          resources_(provider->getResources()),
          ivfConfig_(config),
          minPagedSize_(kMinPageSize) {
    DeviceScope scope(ivfConfig_.device);

    FAISS_THROW_IF_NOT_FMT(
            this->d % 8 == 0,
            "vector dimension (number of bits) "
            "must be divisible by 8 (passed %d)",
            this->d);
    // Distances are carried as floats during k-selection
    FAISS_THROW_IF_NOT_MSG(this->d < (1 << 24), "dimension too large");
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be positive");

    this->is_trained = false;

    quantizer_ = makeQuantizer(resources_, this->d, ivfConfig_);
    resetInvertedLists_();
}

GpuIndexBinaryIVF::~GpuIndexBinaryIVF() {}

int GpuIndexBinaryIVF::getDevice() const {
    return ivfConfig_.device;
}

GpuIndexBinaryFlat* GpuIndexBinaryIVF::getQuantizer() {
    return quantizer_.get();
}

void GpuIndexBinaryIVF::setMinPagingSize(size_t size) {
    minPagedSize_ = size;
}

size_t GpuIndexBinaryIVF::getMinPagingSize() const {
    return minPagedSize_;
}

std::shared_ptr<GpuResources> GpuIndexBinaryIVF::getResources() {
    return resources_;
}

void GpuIndexBinaryIVF::resetInvertedLists_() {
    // destroy old first before allocating new
    index_.reset();
    index_.reset(new BinaryIVF(
            resources_.get(),
            this->d,
            nlist,
            ivfConfig_.indicesOptions,
            ivfConfig_.memorySpace));
}

void GpuIndexBinaryIVF::copyFrom(const faiss::IndexBinaryIVF* index) {
    DeviceScope scope(ivfConfig_.device);

    FAISS_THROW_IF_NOT_FMT(
            index->d % 8 == 0,
            "vector dimension (number of bits) "
            "must be divisible by 8 (passed %d)",
            (int)index->d);
    FAISS_THROW_IF_NOT_MSG(index->d < (1 << 24), "dimension too large");
    FAISS_THROW_IF_NOT_MSG(index->nlist > 0, "nlist must be positive");
    validateNProbe(index->nprobe);

    this->d = index->d;
    this->code_size = index->code_size;
    this->nlist = index->nlist;
    this->nprobe = index->nprobe;
    this->cp = index->cp;
    this->ntotal = index->ntotal;
    this->is_trained = index->is_trained;

    // destroy old state first before allocating new
    quantizer_.reset();
    index_.reset();

    quantizer_ = makeQuantizer(resources_, this->d, ivfConfig_);
    resetInvertedLists_();

    // The other index might not be trained
    if (!index->is_trained) {
        return;
    }

    FAISS_THROW_IF_NOT(index->quantizer);
    FAISS_THROW_IF_NOT_MSG(
            index->quantizer->ntotal == (idx_t)index->nlist,
            "coarse quantizer must hold nlist centroids");

    // Copy the centroids
    std::vector<uint8_t> centroids((size_t)nlist * this->code_size);
    index->quantizer->reconstruct_n(0, nlist, centroids.data());
    quantizer_->add(nlist, centroids.data());

    // Copy all of the IVF data
    index_->copyInvertedListsFrom(index->invlists);
}

void GpuIndexBinaryIVF::copyTo(faiss::IndexBinaryIVF* index) const {
    DeviceScope scope(ivfConfig_.device);

    // We must have the ids in order to copy to the CPU
    FAISS_THROW_IF_NOT_MSG(
            ivfConfig_.indicesOptions != INDICES_IVF,
            "Cannot copy to CPU as GPU index doesn't retain "
            "indices (INDICES_IVF)");

    index->d = this->d;
    index->code_size = this->code_size;
    index->nlist = this->nlist;
    index->nprobe = this->nprobe;
    index->cp = this->cp;
    index->ntotal = this->ntotal;
    index->is_trained = this->is_trained;

    // Replace the quantizer with a CPU flat index holding our centroids
    if (index->own_fields) {
        delete index->quantizer;
    }
    auto flat = new faiss::IndexBinaryFlat(this->d);
    if (this->is_trained) {
        quantizer_->copyTo(flat);
    }
    index->quantizer = flat;
    index->own_fields = true;

    auto ivf = new ArrayInvertedLists(nlist, this->code_size);
    index->replace_invlists(ivf, true);

    if (this->is_trained) {
        index_->copyInvertedListsTo(ivf);
    }
}

void GpuIndexBinaryIVF::train(idx_t n, const uint8_t* x) {
    DeviceScope scope(ivfConfig_.device);

    if (this->is_trained) {
        FAISS_ASSERT(quantizer_->ntotal == nlist);
        return;
    }

    FAISS_THROW_IF_NOT_FMT(
            n >= nlist,
            "not enough training vectors: %ld given for %ld lists",
            n,
            nlist);

    // Run the CPU binary k-means, with the (dominant) assignment step of each
    // iteration done by a GPU flat index, then upload the centroids
    size_t codeSize = this->d / 8;
    faiss::IndexBinaryFlat cpuQuantizer(this->d);
    faiss::IndexBinaryIVF cpuIndex(&cpuQuantizer, this->d, nlist);

    // x may be on the device
    std::vector<uint8_t> hostX;
    if (getDeviceForAddress(x) != -1) {
        auto stream = resources_->getDefaultStream(ivfConfig_.device);
        hostX.resize((size_t)n * codeSize);
        fromDevice(const_cast<uint8_t*>(x), hostX.data(), hostX.size(), stream);
        CUDA_VERIFY(cudaStreamSynchronize(stream));
        x = hostX.data();
    }

    cpuIndex.cp = cp;
    cpuIndex.verbose = verbose;

    GpuIndexFlatConfig clusteringConfig;
    clusteringConfig.device = ivfConfig_.device;
    clusteringConfig.memorySpace = ivfConfig_.memorySpace;
    clusteringConfig.use_cuvs = false;
    GpuResourcesProviderFromInstance provider(resources_);
    GpuIndexFlatL2 clusteringIndex(&provider, this->d, clusteringConfig);
    cpuIndex.clustering_index = &clusteringIndex;

    cpuIndex.train(n, x);
    FAISS_ASSERT(cpuQuantizer.ntotal == nlist);

    quantizer_->reset();
    quantizer_->add(nlist, cpuQuantizer.xb.data());

    this->is_trained = true;
}

void GpuIndexBinaryIVF::add(idx_t n, const uint8_t* x) {
    add_with_ids(n, x, nullptr);
}

void GpuIndexBinaryIVF::add_with_ids(
        idx_t n,
        const uint8_t* x,
        const idx_t* xids) {
    DeviceScope scope(ivfConfig_.device);

    FAISS_THROW_IF_NOT_MSG(this->is_trained, "index must be trained first");

    if (n == 0) {
        return;
    }

    // Page the input so that the host staging buffers stay bounded
    size_t codeSize = this->d / 8;
    idx_t pageSize = std::max<idx_t>(1, minPagedSize_ / codeSize);

    for (idx_t cur = 0; cur < n; cur += pageSize) {
        idx_t num = std::min(pageSize, n - cur);
        addPage_(num, x + (size_t)cur * codeSize, xids ? xids + cur : nullptr);
    }
}

void GpuIndexBinaryIVF::addPage_(idx_t n, const uint8_t* x, const idx_t* xids) {
    size_t codeSize = this->d / 8;

    // We stage the codes through the host to group them by list; they may
    // have been passed on the device
    std::vector<uint8_t> hostX;
    if (getDeviceForAddress(x) != -1) {
        auto stream = resources_->getDefaultStream(ivfConfig_.device);
        hostX.resize((size_t)n * codeSize);
        fromDevice(const_cast<uint8_t*>(x), hostX.data(), hostX.size(), stream);
        CUDA_VERIFY(cudaStreamSynchronize(stream));
        x = hostX.data();
    }

    // The ids may have been passed on the device as well
    std::vector<idx_t> hostIds;
    if (xids && getDeviceForAddress(xids) != -1) {
        auto stream = resources_->getDefaultStream(ivfConfig_.device);
        hostIds.resize(n);
        fromDevice(const_cast<idx_t*>(xids), hostIds.data(), n, stream);
        CUDA_VERIFY(cudaStreamSynchronize(stream));
        xids = hostIds.data();
    }

    // Default ids continue from the current size
    std::vector<idx_t> autoIds;
    if (!xids) {
        autoIds.resize(n);
        std::iota(autoIds.begin(), autoIds.end(), this->ntotal);
        xids = autoIds.data();
    }

    // Find the closest centroid of each vector on the GPU
    std::vector<idx_t> listNos(n);
    quantizer_->assign(n, x, listNos.data());

    this->ntotal += index_->addCodes(x, xids, listNos.data(), n);
}

void GpuIndexBinaryIVF::reset() {
    DeviceScope scope(ivfConfig_.device);

    // Keep the coarse quantizer, clear out the lists
    index_->reset();
    this->ntotal = 0;
}

void GpuIndexBinaryIVF::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    DeviceScope scope(ivfConfig_.device);

    if (n == 0) {
        return;
    }

    FAISS_THROW_IF_NOT_MSG(this->is_trained, "index must be trained first");

    size_t effectiveNprobe = this->nprobe;
    if (params) {
        auto ivfParams = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(ivfParams, "unsupported search parameters");
        FAISS_THROW_IF_NOT_MSG(
                ivfParams->max_codes == 0, "max_codes is not supported");
        FAISS_THROW_IF_NOT_MSG(!ivfParams->sel, "IDSelector not supported");
        effectiveNprobe = ivfParams->nprobe;
    }

    validateKSelect(k);
    validateNProbe(effectiveNprobe);
    effectiveNprobe = std::min(effectiveNprobe, (size_t)nlist);

    // Page the queries so that the temporary buffers stay bounded
    size_t codeSize = this->d / 8;
    idx_t pageSize = n;

    if (getDeviceForAddress(x) == -1) {
        pageSize = std::min(pageSize, (idx_t)(minPagedSize_ / codeSize));
    }

    size_t perQuery = codeSize +
            (effectiveNprobe + 2 * (size_t)k) *
                    (sizeof(int32_t) + sizeof(idx_t));
    pageSize = std::min(
            pageSize,
            (idx_t)(resources_->getTempMemoryAvailableCurrentDevice() / 4 /
                    perQuery));
    pageSize = std::max<idx_t>(1, pageSize);

    for (idx_t cur = 0; cur < n; cur += pageSize) {
        idx_t num = std::min(pageSize, n - cur);
        searchPage_(
                num,
                x + (size_t)cur * codeSize,
                k,
                effectiveNprobe,
                distances + cur * k,
                labels + cur * k);
    }
}

void GpuIndexBinaryIVF::searchPage_(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        size_t effectiveNprobe,
        int32_t* distances,
        idx_t* labels) const {
    auto stream = resources_->getDefaultStream(ivfConfig_.device);

    // Make space for the outputs on the device if they aren't there already
    auto outDistances = toDeviceTemporary<int32_t, 2>(
            resources_.get(), ivfConfig_.device, distances, stream, {n, k});
    auto outIndices = toDeviceTemporary<idx_t, 2>(
            resources_.get(), ivfConfig_.device, labels, stream, {n, k});

    auto queries = toDeviceTemporary<uint8_t, 2>(
            resources_.get(),
            ivfConfig_.device,
            const_cast<uint8_t*>(x),
            stream,
            {n, (idx_t)(this->d / 8)});

    // Find the nprobe closest lists
    DeviceTensor<int32_t, 2, true> coarseDistances(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {n, (idx_t)effectiveNprobe});
    DeviceTensor<idx_t, 2, true> coarseIndices(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {n, (idx_t)effectiveNprobe});

    quantizer_->search(
            n,
            queries.data(),
            effectiveNprobe,
            coarseDistances.data(),
            coarseIndices.data());

    index_->searchBinary(queries, coarseIndices, k, outDistances, outIndices);

    // Copy back if necessary
    fromDevice<int32_t, 2>(outDistances, distances, stream);
    fromDevice<idx_t, 2>(outIndices, labels, stream);
}

idx_t GpuIndexBinaryIVF::getNumLists() const {
    return nlist;
}

idx_t GpuIndexBinaryIVF::getListLength(idx_t listId) const {
    DeviceScope scope(ivfConfig_.device);
    return index_->getListLength(listId);
}

std::vector<uint8_t> GpuIndexBinaryIVF::getListVectorData(idx_t listId) const {
    DeviceScope scope(ivfConfig_.device);
    return index_->getListVectorData(listId, false);
}

std::vector<idx_t> GpuIndexBinaryIVF::getListIndices(idx_t listId) const {
    DeviceScope scope(ivfConfig_.device);
    FAISS_THROW_IF_NOT_MSG(
            ivfConfig_.indicesOptions != INDICES_IVF,
            "ids are not retained (INDICES_IVF)");
    return index_->getListIndices(listId);
}

void GpuIndexBinaryIVF::reserveMemory(size_t numVecs) {
    DeviceScope scope(ivfConfig_.device);
    index_->reserveMemory(numVecs);
}

size_t GpuIndexBinaryIVF::reclaimMemory() {
    DeviceScope scope(ivfConfig_.device);
    return index_->reclaimMemory();
}

} // namespace gpu
} // namespace faiss
