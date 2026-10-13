/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/BinaryIVF.cuh>
#include <faiss/gpu/impl/IVFBinaryScan.cuh>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>

namespace faiss {
namespace gpu {

BinaryIVF::BinaryIVF(
        GpuResources* res,
        int dim,
        idx_t nlist,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IVFBase(res,
                  dim,
                  nlist,
                  faiss::METRIC_L2, // unused; distances are always Hamming
                  0,
                  false, // no residuals
                  false, // not interleaved
                  indicesOptions,
                  space) {
    FAISS_THROW_IF_NOT_FMT(
            dim % 8 == 0,
            "vector dimension (number of bits) must be divisible by 8 "
            "(passed %d)",
            dim);
}

BinaryIVF::~BinaryIVF() {}

size_t BinaryIVF::getCodeSize() const {
    return dim_ / 8;
}

size_t BinaryIVF::getGpuVectorsEncodingSize_(idx_t numVecs) const {
    return (size_t)numVecs * getCodeSize();
}

size_t BinaryIVF::getCpuVectorsEncodingSize_(idx_t numVecs) const {
    return (size_t)numVecs * getCodeSize();
}

std::vector<uint8_t> BinaryIVF::translateCodesToGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const {
    return codes;
}

std::vector<uint8_t> BinaryIVF::translateCodesFromGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const {
    return codes;
}

idx_t BinaryIVF::addCodes(
        const uint8_t* codes,
        const idx_t* ids,
        const idx_t* listNos,
        idx_t n) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    size_t codeSize = getCodeSize();

    // list id -> rows of the input that are appended to that list
    std::map<idx_t, std::vector<idx_t>> rowsByList;
    idx_t numAdded = 0;

    for (idx_t i = 0; i < n; ++i) {
        idx_t listNo = listNos[i];

        // Could not be assigned to a list
        if (listNo < 0) {
            continue;
        }

        FAISS_THROW_IF_NOT_FMT(
                listNo < numLists_,
                "list number %ld is out of bounds (%ld lists total)",
                listNo,
                numLists_);

        rowsByList[listNo].push_back(i);
        ++numAdded;
    }

    if (numAdded == 0) {
        return 0;
    }

    std::vector<idx_t> updatedLists;
    updatedLists.reserve(rowsByList.size());

    for (auto& it : rowsByList) {
        idx_t listNo = it.first;
        auto& rows = it.second;

        // Gather this list's codes and ids contiguously. The host buffers
        // are pageable, so the copies below complete the host read before
        // returning.
        std::vector<uint8_t> listCodes(rows.size() * codeSize);
        std::vector<idx_t> listIds(rows.size());

        for (size_t j = 0; j < rows.size(); ++j) {
            std::memcpy(
                    listCodes.data() + j * codeSize,
                    codes + (size_t)rows[j] * codeSize,
                    codeSize);
            listIds[j] = ids[rows[j]];
        }

        auto& data = deviceListData_[listNo];
        data->data.append(listCodes.data(), listCodes.size(), stream);
        data->numVecs += rows.size();

        auto& indices = deviceListIndices_[listNo];
        if (indicesOptions_ == INDICES_64_BIT) {
            indices->data.append(
                    (const uint8_t*)listIds.data(),
                    listIds.size() * sizeof(idx_t),
                    stream);
            indices->numVecs += rows.size();
            FAISS_ASSERT(indices->numVecs == data->numVecs);
        } else if (indicesOptions_ == INDICES_32_BIT) {
            std::vector<int> listIds32(listIds.size());
            for (size_t j = 0; j < listIds.size(); ++j) {
                FAISS_THROW_IF_NOT_FMT(
                        listIds[j] >= 0 &&
                                listIds[j] <=
                                        (idx_t)std::numeric_limits<int>::max(),
                        "id %ld does not fit in INDICES_32_BIT",
                        listIds[j]);
                listIds32[j] = (int)listIds[j];
            }
            indices->data.append(
                    (const uint8_t*)listIds32.data(),
                    listIds32.size() * sizeof(int),
                    stream);
            indices->numVecs += rows.size();
            FAISS_ASSERT(indices->numVecs == data->numVecs);
        } else if (indicesOptions_ == INDICES_CPU) {
            auto& userIds = listOffsetToUserIndex_[listNo];
            userIds.insert(userIds.end(), listIds.begin(), listIds.end());
            FAISS_ASSERT((idx_t)userIds.size() == data->numVecs);
        } else {
            // INDICES_IVF: the ids are not stored
            FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
        }

        // Used by the multi-pass query to size scratch space
        maxListLength_ = std::max(maxListLength_, data->numVecs);

        updatedLists.push_back(listNo);
    }

    // The base pointers of the lists may have changed with reallocation
    updateDeviceListInfo_(updatedLists, stream);

    return numAdded;
}

void BinaryIVF::searchBinary(
        Tensor<uint8_t, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        int k,
        Tensor<int32_t, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    // These are caught at a higher level
    FAISS_ASSERT(listIds.getSize(1) <= GPU_MAX_SELECTION_K);
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    FAISS_ASSERT(queries.getSize(1) == (idx_t)getCodeSize());
    FAISS_ASSERT(listIds.getSize(0) == queries.getSize(0));
    FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
    FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

    runIVFBinaryScan(
            queries,
            listIds,
            deviceListDataPointers_,
            deviceListIndexPointers_,
            indicesOptions_,
            deviceListLengths_,
            maxListLength_,
            k,
            outDistances,
            outIndices,
            resources_);

    // If the GPU isn't storing the ids, map the (list, offset) results back
    // to user ids on the host
    if (indicesOptions_ == INDICES_CPU) {
        auto stream = resources_->getDefaultStreamCurrentDevice();
        HostTensor<idx_t, 2, true> hostOutIndices(outIndices, stream);

        ivfOffsetToUserIndex(
                hostOutIndices.data(),
                numLists_,
                hostOutIndices.getSize(0),
                hostOutIndices.getSize(1),
                listOffsetToUserIndex_);

        outIndices.copyFrom(hostOutIndices, stream);
    }
}

void BinaryIVF::search(
        Index*,
        Tensor<float, 2, true>&,
        int,
        int,
        Tensor<float, 2, true>&,
        Tensor<idx_t, 2, true>&) {
    FAISS_THROW_MSG(
            "BinaryIVF: float search is not supported; use searchBinary");
}

void BinaryIVF::searchPreassigned(
        Index*,
        Tensor<float, 2, true>&,
        Tensor<float, 2, true>&,
        Tensor<idx_t, 2, true>&,
        int,
        Tensor<float, 2, true>&,
        Tensor<idx_t, 2, true>&,
        bool) {
    FAISS_THROW_MSG(
            "BinaryIVF: float search is not supported; use searchBinary");
}

idx_t BinaryIVF::addVectors(
        Index*,
        Tensor<float, 2, true>&,
        Tensor<idx_t, 1, true>&) {
    FAISS_THROW_MSG("BinaryIVF: float add is not supported; use addCodes");
}

void BinaryIVF::updateQuantizer(Index*) {
    // The coarse quantizer is owned by the GpuIndexBinaryIVF; we keep no
    // copy of the centroids
}

void BinaryIVF::reconstruct_n(idx_t, idx_t, float*) {
    FAISS_THROW_MSG("BinaryIVF: float reconstruct is not supported");
}

void BinaryIVF::appendVectors_(
        Tensor<float, 2, true>&,
        Tensor<float, 2, true>&,
        Tensor<idx_t, 1, true>&,
        Tensor<idx_t, 1, true>&,
        Tensor<idx_t, 1, true>&,
        Tensor<idx_t, 1, true>&,
        Tensor<idx_t, 1, true>&,
        Tensor<idx_t, 1, true>&,
        Tensor<idx_t, 1, true>&,
        cudaStream_t) {
    FAISS_THROW_MSG("BinaryIVF: float add is not supported; use addCodes");
}

} // namespace gpu
} // namespace faiss
