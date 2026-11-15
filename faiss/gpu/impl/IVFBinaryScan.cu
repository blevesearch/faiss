/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/IVFBinaryScan.cuh>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>

#include <algorithm>
#include <limits>

namespace faiss {
namespace gpu {

// Number of threads per block of the Hamming list scan; each thread handles
// a strided subset of the vectors in a list
constexpr int kIVFBinaryScanThreads = 128;

__device__ __forceinline__ int binaryPopcount(uint64_t v) {
    return __popcll(v);
}

__device__ __forceinline__ int binaryPopcount(uint32_t v) {
    return __popc(v);
}

__device__ __forceinline__ int binaryPopcount(uint8_t v) {
    return __popc((uint32_t)v);
}

/// Computes the Hamming distance between the (query, list) pair given by the
/// block, for all vectors in the list. Distances are written as floats (exact
/// for any d < 2^24) so that the generic IVF k-selection passes can be reused.
/// `Word` is the unit in which codes are compared; the code size in bytes
/// must be a multiple of sizeof(Word).
template <typename Word>
__global__ void ivfBinaryScan(
        Tensor<uint8_t, 2, true> queries,
        Tensor<idx_t, 2, true> listIds,
        void** allListData,
        idx_t* listLengths,
        Tensor<idx_t, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance) {
    extern __shared__ __align__(8) unsigned char binaryScanSmem[];

    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x;

    // This is where we start writing out data
    // We ensure that before the array (at offset -1), there is a 0 value
    auto outBase = *(prefixSumOffsets[queryId][probeId].data() - 1);

    idx_t listId = listIds[queryId][probeId];
    // Safety guard in case no list ID was generated
    if (listId == -1) {
        return;
    }

    int codeSize = queries.getSize(1);
    int wordsPerCode = codeSize / sizeof(Word);

    // Stage the query in shared memory, byte-wise as the row itself may not
    // be aligned to Word
    auto queryBytes = queries[queryId].data();
    for (int i = threadIdx.x; i < codeSize; i += blockDim.x) {
        binaryScanSmem[i] = queryBytes[i];
    }
    __syncthreads();

    auto queryWords = reinterpret_cast<const Word*>(binaryScanSmem);
    auto vecs = reinterpret_cast<const Word*>(allListData[listId]);
    idx_t numVecs = listLengths[listId];
    auto distanceOut = distance[outBase].data();

    for (idx_t vec = threadIdx.x; vec < numVecs; vec += blockDim.x) {
        auto code = vecs + vec * wordsPerCode;

        int dist = 0;
        for (int w = 0; w < wordsPerCode; ++w) {
            dist += binaryPopcount((Word)(code[w] ^ queryWords[w]));
        }

        distanceOut[vec] = (float)dist;
    }
}

/// Converts the float distances produced by the selection passes to the
/// int32 Hamming distances that the IndexBinary interface returns
__global__ void ivfBinaryConvertResults(
        const float* floatDistances,
        const idx_t* indices,
        int32_t* outDistances,
        idx_t num) {
    idx_t i = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < num) {
        outDistances[i] = (indices[i] < 0) ? std::numeric_limits<int32_t>::max()
                                           : (int32_t)floatDistances[i];
    }
}

static void runIVFBinaryScanTile(
        Tensor<uint8_t, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        Tensor<char, 1, true>& thrustMem,
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<float, 1, true>& allDistances,
        Tensor<float, 3, true>& heapDistances,
        Tensor<idx_t, 3, true>& heapIndices,
        int k,
        bool use64BitSelection,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res,
        cudaStream_t stream) {
    int codeSize = queries.getSize(1);

    // Calculate offset lengths, so we know where to write out
    // intermediate results
    runCalcListOffsets(
            res, listIds, listLengths, prefixSumOffsets, thrustMem, stream);

    auto grid = dim3(listIds.getSize(1), listIds.getSize(0));
    auto block = dim3(kIVFBinaryScanThreads);
    size_t smem = utils::roundUp(size_t(codeSize), size_t(8));

#define RUN_IVF_BINARY(WORD)                                \
    do {                                                    \
        ivfBinaryScan<WORD><<<grid, block, smem, stream>>>( \
                queries,                                    \
                listIds,                                    \
                listData.data(),                            \
                listLengths.data(),                         \
                prefixSumOffsets,                           \
                allDistances);                              \
    } while (0)

    if (codeSize % 8 == 0) {
        RUN_IVF_BINARY(uint64_t);
    } else if (codeSize % 4 == 0) {
        RUN_IVF_BINARY(uint32_t);
    } else {
        RUN_IVF_BINARY(uint8_t);
    }

    CUDA_TEST_ERROR();

#undef RUN_IVF_BINARY

    // k-select the output in chunks, to increase parallelism; lowest
    // distances are best
    runPass1SelectLists(
            prefixSumOffsets,
            allDistances,
            listIds.getSize(1),
            k,
            use64BitSelection,
            false,
            heapDistances,
            heapIndices,
            stream);

    // k-select final output
    auto flatHeapDistances = heapDistances.downcastInner<2>();
    auto flatHeapIndices = heapIndices.downcastInner<2>();

    runPass2SelectLists(
            flatHeapDistances,
            flatHeapIndices,
            listIndices,
            indicesOptions,
            prefixSumOffsets,
            listIds,
            k,
            use64BitSelection,
            false,
            outDistances,
            outIndices,
            stream);
}

void runIVFBinaryScan(
        Tensor<uint8_t, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
        int k,
        Tensor<int32_t, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res) {
    auto stream = res->getDefaultStreamCurrentDevice();

    auto nq = queries.getSize(0);
    auto nprobe = listIds.getSize(1);

    // Nothing is stored, so there is nothing to scan. Avoids zero-sized
    // temporary allocations below.
    if (nq == 0 || maxListLength == 0) {
        auto numResults = outDistances.numElements();
        if (numResults > 0) {
            // All indices are -1 (0xff bytes); the conversion kernel maps
            // those to INT32_MAX without reading the float distances
            CUDA_VERIFY(cudaMemsetAsync(
                    outIndices.data(),
                    0xff,
                    numResults * sizeof(idx_t),
                    stream));
            ivfBinaryConvertResults<<<
                    utils::divUp(numResults, idx_t(256)),
                    256,
                    0,
                    stream>>>(
                    nullptr,
                    outIndices.data(),
                    outDistances.data(),
                    numResults);
            CUDA_TEST_ERROR();
        }
        return;
    }

    // If the maximum list length (in terms of number of vectors) times nprobe
    // (number of lists) is > 2^31 - 1, then we will use 64-bit indexing in the
    // selection kernels
    bool use64BitSelection =
            maxListLength * nprobe > idx_t(std::numeric_limits<int32_t>::max());

    // Make a reservation for Thrust to do its dirty work (global memory
    // cross-block reduction space); hopefully this is large enough.
    constexpr idx_t kThrustMemSize = 16384;

    DeviceTensor<char, 1, true> thrustMem1(
            res, makeTempAlloc(AllocType::Other, stream), {kThrustMemSize});
    DeviceTensor<char, 1, true> thrustMem2(
            res, makeTempAlloc(AllocType::Other, stream), {kThrustMemSize});
    DeviceTensor<char, 1, true>* thrustMem[2] = {&thrustMem1, &thrustMem2};

    // How much temporary memory would we need to handle a single query?
    size_t sizePerQuery = getIVFPerQueryTempMemory(k, nprobe, maxListLength);

    // How many queries do we wish to run at once?
    idx_t queryTileSize = getIVFQueryTileSize(
            nq, res->getTempMemoryAvailableCurrentDevice(), sizePerQuery);

    // Temporary memory buffers
    // Make sure there is space prior to the start which will be 0, and
    // will handle the boundary condition without branches
    DeviceTensor<idx_t, 1, true> prefixSumOffsetSpace1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe + 1});
    DeviceTensor<idx_t, 1, true> prefixSumOffsetSpace2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe + 1});

    DeviceTensor<idx_t, 2, true> prefixSumOffsets1(
            prefixSumOffsetSpace1[1].data(), {queryTileSize, nprobe});
    DeviceTensor<idx_t, 2, true> prefixSumOffsets2(
            prefixSumOffsetSpace2[1].data(), {queryTileSize, nprobe});
    DeviceTensor<idx_t, 2, true>* prefixSumOffsets[2] = {
            &prefixSumOffsets1, &prefixSumOffsets2};

    // Make sure the element before prefixSumOffsets is 0, since we
    // depend upon simple, boundary-less indexing to get proper results
    CUDA_VERIFY(cudaMemsetAsync(
            prefixSumOffsetSpace1.data(), 0, sizeof(idx_t), stream));
    CUDA_VERIFY(cudaMemsetAsync(
            prefixSumOffsetSpace2.data(), 0, sizeof(idx_t), stream));

    DeviceTensor<float, 1, true> allDistances1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe * maxListLength});
    DeviceTensor<float, 1, true> allDistances2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe * maxListLength});
    DeviceTensor<float, 1, true>* allDistances[2] = {
            &allDistances1, &allDistances2};

    idx_t pass2Chunks = getIVFKSelectionPass2Chunks(nprobe);
    DeviceTensor<float, 3, true> heapDistances1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<float, 3, true> heapDistances2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<float, 3, true>* heapDistances[2] = {
            &heapDistances1, &heapDistances2};

    DeviceTensor<idx_t, 3, true> heapIndices1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<idx_t, 3, true> heapIndices2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<idx_t, 3, true>* heapIndices[2] = {
            &heapIndices1, &heapIndices2};

    // The selection passes produce float distances; we convert at the end
    DeviceTensor<float, 2, true> floatDistances(
            res, makeTempAlloc(AllocType::Other, stream), {nq, (idx_t)k});

    auto streams = res->getAlternateStreamsCurrentDevice();
    res->streamWait(streams, {stream});

    int curStream = 0;

    for (idx_t query = 0; query < nq; query += queryTileSize) {
        auto numQueriesInTile = std::min(queryTileSize, nq - query);

        auto prefixSumOffsetsView =
                prefixSumOffsets[curStream]->narrowOutermost(
                        0, numQueriesInTile);

        auto listIdsView = listIds.narrowOutermost(query, numQueriesInTile);
        auto queryView = queries.narrowOutermost(query, numQueriesInTile);

        auto heapDistancesView =
                heapDistances[curStream]->narrowOutermost(0, numQueriesInTile);
        auto heapIndicesView =
                heapIndices[curStream]->narrowOutermost(0, numQueriesInTile);

        auto outDistanceView =
                floatDistances.narrowOutermost(query, numQueriesInTile);
        auto outIndicesView =
                outIndices.narrowOutermost(query, numQueriesInTile);

        runIVFBinaryScanTile(
                queryView,
                listIdsView,
                listData,
                listIndices,
                indicesOptions,
                listLengths,
                *thrustMem[curStream],
                prefixSumOffsetsView,
                *allDistances[curStream],
                heapDistancesView,
                heapIndicesView,
                k,
                use64BitSelection,
                outDistanceView,
                outIndicesView,
                res,
                streams[curStream]);

        curStream = (curStream + 1) % 2;
    }

    res->streamWait({stream}, streams);

    auto numResults = outDistances.numElements();
    auto blocks = utils::divUp(numResults, idx_t(256));
    ivfBinaryConvertResults<<<blocks, 256, 0, stream>>>(
            floatDistances.data(),
            outIndices.data(),
            outDistances.data(),
            numResults);
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
