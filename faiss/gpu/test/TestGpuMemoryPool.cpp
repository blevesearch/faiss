/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/MemoryPool.h>
#include <gtest/gtest.h>

// Exercises temporary-memory overflow through an externally managed GpuMemoryPool.
TEST(TestGpuMemoryPool, TempOverflowViaPool) {
    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    const size_t kTempMem = 64 * 1024;
    const size_t kPoolCap = 64 * 1024 * 1024;

    faiss::gpu::GpuMemoryPool pool(device, kPoolCap);

    faiss::gpu::StandardGpuResources res;
    res.setTempMemory(kTempMem);
    res.setTempMemoryOverflowPool(&pool);

    const int dim = 128;
    const int numVecs = 10000;
    const int numQuery = 10;
    const int k = 10;

    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    config.use_cuvs = false;

    faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, dim, config);
    faiss::IndexFlatL2 cpuIndex(dim);

    auto vecs = faiss::gpu::randVecs(numVecs, dim);
    gpuIndex.add(numVecs, vecs.data());
    cpuIndex.add(numVecs, vecs.data());

    auto queries = faiss::gpu::randVecs(numQuery, dim);
    compareIndices(
            queries,
            cpuIndex,
            gpuIndex,
            numQuery,
            dim,
            k,
            "",
            6e-3f,
            0.1f,
            0.015f);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
