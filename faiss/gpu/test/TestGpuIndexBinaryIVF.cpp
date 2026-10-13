/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexBinaryFlat.h>
#include <faiss/gpu/GpuIndexBinaryIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace {

std::vector<uint8_t> randomCodes(size_t n, size_t codeSize, int seed) {
    std::vector<uint8_t> codes(n * codeSize);
    unsigned state = seed;
    for (auto& c : codes) {
        state = state * 1664525u + 1013904223u;
        c = (uint8_t)(state >> 24);
    }
    return codes;
}

} // namespace

// Builds a CPU index, clones it to the GPU, and checks that both return the
// same Hamming distances for the same nprobe
void testCopyFromCpu(int dims, int nlist, int nprobe, int k) {
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    int codeSize = dims / 8;
    int numTrain = nlist * 40;
    int numAdd = 10000;
    int numQuery = 100;

    auto train = randomCodes(numTrain, codeSize, 1);
    auto add = randomCodes(numAdd, codeSize, 2);
    auto query = randomCodes(numQuery, codeSize, 3);

    faiss::IndexBinaryFlat quantizer(dims);
    faiss::IndexBinaryIVF cpuIndex(&quantizer, dims, nlist);
    cpuIndex.nprobe = nprobe;
    cpuIndex.train(numTrain, train.data());
    cpuIndex.add(numAdd, add.data());

    faiss::gpu::GpuIndexBinaryIVFConfig config;
    config.device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    faiss::gpu::GpuIndexBinaryIVF gpuIndex(&res, &cpuIndex, config);

    EXPECT_EQ(gpuIndex.ntotal, cpuIndex.ntotal);
    EXPECT_EQ(gpuIndex.nprobe, (size_t)nprobe);
    for (int i = 0; i < nlist; ++i) {
        EXPECT_EQ(gpuIndex.getListLength(i), cpuIndex.get_list_size(i));
    }

    std::vector<int32_t> cpuDist(numQuery * k), gpuDist(numQuery * k);
    std::vector<faiss::idx_t> cpuLabels(numQuery * k), gpuLabels(numQuery * k);
    cpuIndex.search(
            numQuery, query.data(), k, cpuDist.data(), cpuLabels.data());
    gpuIndex.search(
            numQuery, query.data(), k, gpuDist.data(), gpuLabels.data());

    // With partial probing, ties in the coarse quantizer distances (common
    // for short codes) may select different lists on the CPU and GPU, so
    // results can legitimately differ. We check that every returned label is
    // at its reported distance below, and compare exactly with exhaustive
    // probing further down.
    for (int i = 0; i < numQuery; ++i) {
        for (int j = 0; j < k; ++j) {
            if (gpuLabels[i * k + j] < 0) {
                EXPECT_EQ(cpuLabels[i * k + j], -1);
                continue;
            }
            // the label must truly be at the reported distance
            int dist = 0;
            for (int b = 0; b < codeSize; ++b) {
                dist += __builtin_popcount(
                        query[i * codeSize + b] ^
                        add[gpuLabels[i * k + j] * codeSize + b]);
            }
            EXPECT_EQ(dist, gpuDist[i * k + j]);
        }
    }

    // Probing every list is exhaustive, so the results must match exactly
    // (modulo the order of labels with equal distances)
    {
        cpuIndex.nprobe = nlist;
        gpuIndex.nprobe = nlist;
        std::vector<int32_t> cd(numQuery * k), gd(numQuery * k);
        std::vector<faiss::idx_t> cl(numQuery * k), gl(numQuery * k);
        cpuIndex.search(numQuery, query.data(), k, cd.data(), cl.data());
        gpuIndex.search(numQuery, query.data(), k, gd.data(), gl.data());
        EXPECT_EQ(cd, gd);
        cpuIndex.nprobe = nprobe;
        gpuIndex.nprobe = nprobe;
    }

    // Round trip back to the CPU
    faiss::IndexBinaryIVF back;
    gpuIndex.copyTo(&back);
    EXPECT_EQ(back.ntotal, cpuIndex.ntotal);
    EXPECT_EQ(back.nlist, cpuIndex.nlist);
    for (int i = 0; i < nlist; ++i) {
        EXPECT_EQ(back.get_list_size(i), cpuIndex.get_list_size(i));
    }
    EXPECT_EQ(back.nprobe, (size_t)nprobe);
    std::vector<int32_t> backDist(numQuery * k);
    std::vector<faiss::idx_t> backLabels(numQuery * k);
    back.search(numQuery, query.data(), k, backDist.data(), backLabels.data());
    EXPECT_EQ(cpuDist, backDist);
}

TEST(TestGpuIndexBinaryIVF, CopyFromCpu_64) {
    testCopyFromCpu(64, 16, 4, 10);
}

TEST(TestGpuIndexBinaryIVF, CopyFromCpu_128) {
    testCopyFromCpu(128, 32, 8, 20);
}

// codeSize multiple of 4 but not 8, and not a multiple of 4
TEST(TestGpuIndexBinaryIVF, CopyFromCpu_96) {
    testCopyFromCpu(96, 16, 16, 5);
}

TEST(TestGpuIndexBinaryIVF, CopyFromCpu_40) {
    testCopyFromCpu(40, 16, 3, 5);
}

// Training, adding with ids and searching on the GPU only
TEST(TestGpuIndexBinaryIVF, TrainAddSearch) {
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    int dims = 256;
    int codeSize = dims / 8;
    int nlist = 16;
    int numAdd = 5000;
    int numQuery = 50;
    int k = 8;

    auto data = randomCodes(numAdd, codeSize, 7);
    auto query = randomCodes(numQuery, codeSize, 8);
    std::vector<faiss::idx_t> ids(numAdd);
    for (int i = 0; i < numAdd; ++i) {
        ids[i] = 1000000 + i;
    }

    faiss::gpu::GpuIndexBinaryIVF gpuIndex(&res, dims, nlist);
    EXPECT_FALSE(gpuIndex.is_trained);
    gpuIndex.train(numAdd, data.data());
    EXPECT_TRUE(gpuIndex.is_trained);

    gpuIndex.add_with_ids(numAdd, data.data(), ids.data());
    EXPECT_EQ(gpuIndex.ntotal, numAdd);

    faiss::idx_t total = 0;
    for (int i = 0; i < nlist; ++i) {
        total += gpuIndex.getListLength(i);
    }
    EXPECT_EQ(total, numAdd);

    // Probing every list is exhaustive, so must match brute force
    gpuIndex.nprobe = nlist;
    faiss::IndexBinaryFlat flat(dims);
    flat.add(numAdd, data.data());

    std::vector<int32_t> flatDist(numQuery * k), gpuDist(numQuery * k);
    std::vector<faiss::idx_t> flatLabels(numQuery * k), gpuLabels(numQuery * k);
    flat.search(numQuery, query.data(), k, flatDist.data(), flatLabels.data());
    gpuIndex.search(
            numQuery, query.data(), k, gpuDist.data(), gpuLabels.data());

    EXPECT_EQ(flatDist, gpuDist);
    for (int i = 0; i < numQuery * k; ++i) {
        EXPECT_EQ(gpuLabels[i] - 1000000 >= 0, true);
    }

    // Adding more vectors without ids continues from ntotal
    gpuIndex.add(10, data.data());
    EXPECT_EQ(gpuIndex.ntotal, numAdd + 10);

    gpuIndex.reset();
    EXPECT_EQ(gpuIndex.ntotal, 0);
    EXPECT_TRUE(gpuIndex.is_trained);
    gpuIndex.search(
            numQuery, query.data(), k, gpuDist.data(), gpuLabels.data());
    for (int i = 0; i < numQuery * k; ++i) {
        EXPECT_EQ(gpuLabels[i], -1);
        EXPECT_EQ(gpuDist[i], std::numeric_limits<int32_t>::max());
    }
}

namespace {

// Trained and filled CPU index plus the data used to build it
struct CpuFixture {
    int dims = 128;
    int codeSize = 16;
    int nlist = 16;
    int numAdd = 4000;
    int numQuery = 40;
    std::vector<uint8_t> train, add, query;
    faiss::IndexBinaryFlat quantizer;
    faiss::IndexBinaryIVF index;

    CpuFixture() : quantizer(128), index(&quantizer, 128, 16) {
        train = randomCodes(nlist * 40, codeSize, 11);
        add = randomCodes(numAdd, codeSize, 12);
        query = randomCodes(numQuery, codeSize, 13);
        index.train(nlist * 40, train.data());
        index.add(numAdd, add.data());
    }
};

} // namespace

// x and ids resident on the device must work like host-resident ones
TEST(TestGpuIndexBinaryIVF, DeviceInputs) {
    faiss::gpu::StandardGpuResources res;
    CpuFixture f;
    int k = 5;

    faiss::gpu::GpuIndexBinaryIVF host(&res, f.dims, f.nlist);
    faiss::gpu::GpuIndexBinaryIVF dev(&res, f.dims, f.nlist);
    host.train(f.nlist * 40, f.train.data());
    dev.copyFrom(&f.index); // trained, empty copy of the centroids
    dev.reset();
    host.copyFrom(&f.index);
    host.reset();

    std::vector<faiss::idx_t> ids(f.numAdd);
    for (int i = 0; i < f.numAdd; ++i) {
        ids[i] = 5000 + 3 * i;
    }

    uint8_t* dx = nullptr;
    faiss::idx_t* dids = nullptr;
    ASSERT_EQ(cudaMalloc(&dx, f.add.size()), cudaSuccess);
    ASSERT_EQ(
            cudaMalloc(&dids, ids.size() * sizeof(faiss::idx_t)), cudaSuccess);
    cudaMemcpy(dx, f.add.data(), f.add.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(
            dids,
            ids.data(),
            ids.size() * sizeof(faiss::idx_t),
            cudaMemcpyHostToDevice);

    host.add_with_ids(f.numAdd, f.add.data(), ids.data());
    dev.add_with_ids(f.numAdd, dx, dids);
    EXPECT_EQ(host.ntotal, dev.ntotal);

    host.nprobe = dev.nprobe = f.nlist;
    std::vector<int32_t> hd(f.numQuery * k), dd(f.numQuery * k);
    std::vector<faiss::idx_t> hl(f.numQuery * k), dl(f.numQuery * k);
    host.search(f.numQuery, f.query.data(), k, hd.data(), hl.data());
    dev.search(f.numQuery, f.query.data(), k, dd.data(), dl.data());
    EXPECT_EQ(hd, dd);
    for (auto l : dl) {
        EXPECT_GE(l, 5000);
        EXPECT_EQ((l - 5000) % 3, 0);
    }

    // list contents carry the user ids
    for (int i = 0; i < f.nlist; ++i) {
        EXPECT_EQ(dev.getListLength(i), dev.getListIndices(i).size());
        EXPECT_EQ(
                dev.getListVectorData(i).size(),
                dev.getListLength(i) * f.codeSize);
    }

    cudaFree(dx);
    cudaFree(dids);
}

// Asking for more results than there are vectors pads with label -1
TEST(TestGpuIndexBinaryIVF, KLargerThanNtotal) {
    faiss::gpu::StandardGpuResources res;
    CpuFixture f;
    faiss::gpu::GpuIndexBinaryIVF gpu(&res, f.dims, f.nlist);
    gpu.train(f.nlist * 40, f.train.data());
    gpu.add(3, f.add.data());
    gpu.nprobe = f.nlist;

    int k = 10;
    std::vector<int32_t> d(f.numQuery * k);
    std::vector<faiss::idx_t> l(f.numQuery * k);
    gpu.search(f.numQuery, f.query.data(), k, d.data(), l.data());
    for (int i = 0; i < f.numQuery; ++i) {
        for (int j = 0; j < k; ++j) {
            if (j < 3) {
                EXPECT_GE(l[i * k + j], 0);
                EXPECT_LT(l[i * k + j], 3);
            } else {
                EXPECT_EQ(l[i * k + j], -1);
                EXPECT_EQ(d[i * k + j], std::numeric_limits<int32_t>::max());
            }
        }
    }
}

// nprobe given through search parameters overrides the index's nprobe, and
// nprobe larger than nlist is clamped
TEST(TestGpuIndexBinaryIVF, SearchParams) {
    faiss::gpu::StandardGpuResources res;
    CpuFixture f;
    faiss::gpu::GpuIndexBinaryIVF gpu(&res, &f.index);
    int k = 5;

    std::vector<int32_t> a(f.numQuery * k), b(f.numQuery * k),
            c(f.numQuery * k);
    std::vector<faiss::idx_t> l(f.numQuery * k);

    gpu.nprobe = f.nlist;
    gpu.search(f.numQuery, f.query.data(), k, a.data(), l.data());

    gpu.nprobe = 1;
    faiss::IVFSearchParameters params;
    params.nprobe = f.nlist;
    gpu.search(f.numQuery, f.query.data(), k, b.data(), l.data(), &params);
    EXPECT_EQ(a, b);

    params.nprobe = f.nlist * 4;
    gpu.search(f.numQuery, f.query.data(), k, c.data(), l.data(), &params);
    EXPECT_EQ(a, c);

    // out of range for GPU selection
    params.nprobe = 1000000;
    EXPECT_THROW(
            gpu.search(
                    f.numQuery, f.query.data(), k, c.data(), l.data(), &params),
            faiss::FaissException);

    params.nprobe = 1;
    params.max_codes = 10;
    EXPECT_THROW(
            gpu.search(
                    f.numQuery, f.query.data(), k, c.data(), l.data(), &params),
            faiss::FaissException);
}

// Copying an untrained CPU index gives an untrained GPU index
TEST(TestGpuIndexBinaryIVF, CopyFromUntrained) {
    faiss::gpu::StandardGpuResources res;
    CpuFixture f;
    faiss::IndexBinaryFlat q(f.dims);
    faiss::IndexBinaryIVF untrained(&q, f.dims, f.nlist);

    faiss::gpu::GpuIndexBinaryIVF gpu(&res, &untrained);
    EXPECT_FALSE(gpu.is_trained);
    EXPECT_THROW(gpu.add(10, f.add.data()), faiss::FaissException);

    std::vector<int32_t> d(5);
    std::vector<faiss::idx_t> l(5);
    EXPECT_THROW(
            gpu.search(1, f.query.data(), 5, d.data(), l.data()),
            faiss::FaissException);

    gpu.train(f.nlist * 40, f.train.data());
    EXPECT_TRUE(gpu.is_trained);
    gpu.add(f.numAdd, f.add.data());
    EXPECT_EQ(gpu.ntotal, f.numAdd);

    // and an untrained GPU index copies back as untrained
    faiss::gpu::GpuIndexBinaryIVF fresh(&res, f.dims, f.nlist);
    faiss::IndexBinaryIVF back;
    fresh.copyTo(&back);
    EXPECT_FALSE(back.is_trained);
    EXPECT_EQ(back.ntotal, 0);
}

TEST(TestGpuIndexBinaryIVF, ReserveAndReclaim) {
    faiss::gpu::StandardGpuResources res;
    CpuFixture f;
    faiss::gpu::GpuIndexBinaryIVF gpu(&res, &f.index);
    int k = 5;
    gpu.nprobe = f.nlist;

    std::vector<int32_t> before(f.numQuery * k), after(f.numQuery * k);
    std::vector<faiss::idx_t> l(f.numQuery * k);
    gpu.search(f.numQuery, f.query.data(), k, before.data(), l.data());

    gpu.reset();
    gpu.reserveMemory(100000);
    gpu.add(f.numAdd, f.add.data());
    gpu.reclaimMemory();
    gpu.search(f.numQuery, f.query.data(), k, after.data(), l.data());
    EXPECT_EQ(before, after);
}

// Cloning through the generic binary cloner entry points
TEST(TestGpuIndexBinaryIVF, Cloner) {
    faiss::gpu::StandardGpuResources res;
    CpuFixture f;
    int k = 5;

    faiss::gpu::GpuClonerOptions options;
    std::unique_ptr<faiss::IndexBinary> gpu(
            faiss::gpu::index_binary_cpu_to_gpu(&res, 0, &f.index, &options));
    auto gpuIvf = dynamic_cast<faiss::gpu::GpuIndexBinaryIVF*>(gpu.get());
    ASSERT_NE(gpuIvf, nullptr);
    EXPECT_EQ(gpuIvf->ntotal, f.index.ntotal);

    std::unique_ptr<faiss::IndexBinary> back(
            faiss::gpu::index_binary_gpu_to_cpu(gpu.get()));
    auto cpuIvf = dynamic_cast<faiss::IndexBinaryIVF*>(back.get());
    ASSERT_NE(cpuIvf, nullptr);
    EXPECT_EQ(cpuIvf->ntotal, f.index.ntotal);
    EXPECT_EQ(cpuIvf->nlist, f.index.nlist);

    cpuIvf->nprobe = f.nlist;
    f.index.nprobe = f.nlist;
    std::vector<int32_t> a(f.numQuery * k), b(f.numQuery * k);
    std::vector<faiss::idx_t> l(f.numQuery * k);
    f.index.search(f.numQuery, f.query.data(), k, a.data(), l.data());
    cpuIvf->search(f.numQuery, f.query.data(), k, b.data(), l.data());
    EXPECT_EQ(a, b);

    // index storage options are honored
    options.indicesOptions = faiss::gpu::INDICES_32_BIT;
    std::unique_ptr<faiss::IndexBinary> gpu32(
            faiss::gpu::index_binary_cpu_to_gpu(&res, 0, &f.index, &options));
    EXPECT_EQ(gpu32->ntotal, f.index.ntotal);

    // reserveVecs is honored for empty indexes
    faiss::IndexBinaryFlat q(f.dims);
    faiss::IndexBinaryIVF empty(&q, f.dims, f.nlist);
    empty.train(f.nlist * 40, f.train.data());
    faiss::gpu::GpuClonerOptions reserve;
    reserve.reserveVecs = 10000;
    std::unique_ptr<faiss::IndexBinary> reserved(
            faiss::gpu::index_binary_cpu_to_gpu(&res, 0, &empty, &reserve));
    EXPECT_EQ(reserved->ntotal, 0);
}

namespace {

int hamming(const uint8_t* a, const uint8_t* b, size_t codeSize) {
    int d = 0;
    for (size_t i = 0; i < codeSize; ++i) {
        d += __builtin_popcount((unsigned)(a[i] ^ b[i]));
    }
    return d;
}

} // namespace

// Every supported way of storing the ids must return the same results
TEST(TestGpuIndexBinaryIVF, IndicesOptions) {
    faiss::gpu::StandardGpuResources res;
    CpuFixture f;
    int k = 6;
    f.index.nprobe = f.nlist;

    std::vector<int32_t> refDist(f.numQuery * k);
    std::vector<faiss::idx_t> refLabels(f.numQuery * k);
    f.index.search(
            f.numQuery, f.query.data(), k, refDist.data(), refLabels.data());

    for (auto opt :
         {faiss::gpu::INDICES_64_BIT,
          faiss::gpu::INDICES_32_BIT,
          faiss::gpu::INDICES_CPU,
          faiss::gpu::INDICES_IVF}) {
        faiss::gpu::GpuIndexBinaryIVFConfig config;
        config.indicesOptions = opt;
        faiss::gpu::GpuIndexBinaryIVF gpu(&res, &f.index, config);
        gpu.nprobe = f.nlist;

        std::vector<int32_t> dist(f.numQuery * k);
        std::vector<faiss::idx_t> labels(f.numQuery * k);
        gpu.search(f.numQuery, f.query.data(), k, dist.data(), labels.data());
        EXPECT_EQ(dist, refDist) << "options " << opt;

        if (opt == faiss::gpu::INDICES_IVF) {
            // the ids are replaced by (list, offset) pairs
            for (auto l : labels) {
                EXPECT_GE(l, 0);
            }
            faiss::IndexBinaryFlat q(f.dims);
            faiss::IndexBinaryIVF cpu(&q, f.dims, f.nlist);
            EXPECT_THROW(gpu.copyTo(&cpu), faiss::FaissException);
            EXPECT_THROW(gpu.getListIndices(0), faiss::FaissException);
            continue;
        }

        // the returned ids must be at the reported distances
        for (int i = 0; i < f.numQuery; ++i) {
            for (int j = 0; j < k; ++j) {
                auto id = labels[i * k + j];
                ASSERT_GE(id, 0);
                EXPECT_EQ(
                        hamming(f.query.data() + i * f.codeSize,
                                f.add.data() + id * f.codeSize,
                                f.codeSize),
                        dist[i * k + j]);
            }
        }

        // adding vectors with custom ids, and the round trip to the CPU
        std::vector<faiss::idx_t> ids(100);
        for (int i = 0; i < 100; ++i) {
            ids[i] = 777000 + i;
        }
        gpu.add_with_ids(100, f.add.data(), ids.data());
        faiss::IndexBinaryFlat q(f.dims);
        faiss::IndexBinaryIVF cpu(&q, f.dims, f.nlist);
        gpu.copyTo(&cpu);
        EXPECT_EQ(cpu.ntotal, f.index.ntotal + 100);
        faiss::idx_t custom = 0;
        for (int l = 0; l < f.nlist; ++l) {
            for (size_t j = 0; j < cpu.get_list_size(l); ++j) {
                custom += cpu.invlists->get_single_id(l, j) >= 777000;
            }
        }
        EXPECT_EQ(custom, 100);
    }

    // ids that cannot be represented are rejected
    faiss::gpu::GpuIndexBinaryIVFConfig config;
    config.indicesOptions = faiss::gpu::INDICES_32_BIT;
    faiss::gpu::GpuIndexBinaryIVF gpu(&res, &f.index, config);
    faiss::idx_t big = (faiss::idx_t)1 << 40;
    EXPECT_THROW(
            gpu.add_with_ids(1, f.add.data(), &big), faiss::FaissException);
}

// Adding and searching in tiny pages must not change the results
TEST(TestGpuIndexBinaryIVF, Paging) {
    faiss::gpu::StandardGpuResources res;
    CpuFixture f;
    int k = 7;

    faiss::gpu::GpuIndexBinaryIVF whole(&res, f.dims, f.nlist);
    faiss::gpu::GpuIndexBinaryIVF paged(&res, f.dims, f.nlist);
    paged.setMinPagingSize(100);
    EXPECT_EQ(paged.getMinPagingSize(), 100);

    whole.train(f.nlist * 40, f.train.data());
    paged.train(f.nlist * 40, f.train.data());
    // identical centroids so the lists are comparable
    std::vector<uint8_t> centroids(f.nlist * f.codeSize);
    whole.getQuantizer()->reconstruct_n(0, f.nlist, centroids.data());
    paged.getQuantizer()->reset();
    paged.getQuantizer()->add(f.nlist, centroids.data());

    whole.add(f.numAdd, f.add.data());
    paged.add(f.numAdd, f.add.data());
    EXPECT_EQ(paged.ntotal, f.numAdd);
    for (int l = 0; l < f.nlist; ++l) {
        EXPECT_EQ(whole.getListIndices(l), paged.getListIndices(l));
    }

    whole.nprobe = paged.nprobe = 4;
    std::vector<int32_t> d1(f.numQuery * k), d2(f.numQuery * k);
    std::vector<faiss::idx_t> l1(f.numQuery * k), l2(f.numQuery * k);
    whole.search(f.numQuery, f.query.data(), k, d1.data(), l1.data());
    paged.search(f.numQuery, f.query.data(), k, d2.data(), l2.data());
    EXPECT_EQ(d1, d2);
    EXPECT_EQ(l1, l2);
}

// Training honors the clustering parameters and exposes the quantizer
TEST(TestGpuIndexBinaryIVF, TrainClusteringParams) {
    faiss::gpu::StandardGpuResources res;
    CpuFixture f;

    faiss::gpu::GpuIndexBinaryIVF gpu(&res, f.dims, f.nlist);
    gpu.cp.niter = 3;
    gpu.cp.seed = 42;
    gpu.train(f.nlist * 40, f.train.data());
    EXPECT_TRUE(gpu.is_trained);
    EXPECT_EQ(gpu.getQuantizer()->ntotal, f.nlist);

    // same parameters give the same centroids
    faiss::gpu::GpuIndexBinaryIVF again(&res, f.dims, f.nlist);
    again.cp = gpu.cp;
    again.train(f.nlist * 40, f.train.data());
    std::vector<uint8_t> c1(f.nlist * f.codeSize), c2(c1.size());
    gpu.getQuantizer()->reconstruct_n(0, f.nlist, c1.data());
    again.getQuantizer()->reconstruct_n(0, f.nlist, c2.data());
    EXPECT_EQ(c1, c2);

    // the parameters survive the CPU round trip
    faiss::IndexBinaryFlat q(f.dims);
    faiss::IndexBinaryIVF cpu(&q, f.dims, f.nlist);
    gpu.copyTo(&cpu);
    EXPECT_EQ(cpu.cp.niter, 3);
    EXPECT_EQ(cpu.cp.seed, 42);
}

// Sharding an IVF index over several GPUs
TEST(TestGpuIndexBinaryIVF, MultiGpuShard) {
    int numDevices = faiss::gpu::getNumDevices();
    if (numDevices < 2) {
        GTEST_SKIP() << "needs at least 2 GPUs";
    }

    CpuFixture f;
    int k = 6;
    f.index.nprobe = f.nlist;

    std::vector<std::unique_ptr<faiss::gpu::StandardGpuResources>> owned;
    std::vector<faiss::gpu::GpuResourcesProvider*> providers;
    std::vector<int> devices;
    for (int i = 0; i < 2; ++i) {
        owned.emplace_back(new faiss::gpu::StandardGpuResources);
        providers.push_back(owned.back().get());
        devices.push_back(i);
    }

    faiss::gpu::GpuMultipleClonerOptions options;
    options.shard = true;
    std::unique_ptr<faiss::IndexBinary> sharded(
            faiss::gpu::index_binary_cpu_to_gpu_multiple(
                    providers, devices, &f.index, &options));
    EXPECT_EQ(sharded->ntotal, f.index.ntotal);

    std::vector<int32_t> refDist(f.numQuery * k), dist(f.numQuery * k);
    std::vector<faiss::idx_t> refLabels(f.numQuery * k), labels(f.numQuery * k);
    f.index.search(
            f.numQuery, f.query.data(), k, refDist.data(), refLabels.data());
    sharded->search(f.numQuery, f.query.data(), k, dist.data(), labels.data());
    EXPECT_EQ(dist, refDist);
    for (int i = 0; i < f.numQuery * k; ++i) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], f.numAdd);
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
