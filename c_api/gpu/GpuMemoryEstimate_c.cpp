/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "GpuMemoryEstimate_c.h"

#include <cstdint>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/invlists/InvertedLists.h>

#include "macros_impl.h"

namespace {

using faiss::Index;
using faiss::IndexFlat;
using faiss::IndexIDMap;
using faiss::IndexIVF;
using faiss::IndexIVFFlat;
using faiss::IndexIVFPQ;
using faiss::IndexIVFScalarQuantizer;
using faiss::IndexPreTransform;
using faiss::IndexScalarQuantizer;
using faiss::ScalarQuantizer;
using faiss::gpu::GpuClonerOptions;
using faiss::gpu::IndicesOptions;

// Size in bytes used per stored index entry given the user's IndicesOptions.
inline size_t indices_entry_bytes(IndicesOptions opts) {
    switch (opts) {
        case faiss::gpu::INDICES_32_BIT:
            return sizeof(int32_t);
        case faiss::gpu::INDICES_64_BIT:
            return sizeof(int64_t);
        default:
            // INDICES_CPU / INDICES_IVF / INDICES_NONE do not store
            // per-vector indices on the GPU.
            return 0;
    }
}

// Size in bytes of one element of the coarse quantizer's centroid storage.
inline size_t centroid_element_bytes(const GpuClonerOptions& opts) {
    // half (fp16) is 2 bytes, float is 4 bytes.
    return opts.useFloat16CoarseQuantizer ? sizeof(uint16_t) : sizeof(float);
}

// Number of interleaved blocks of warp_size vectors required to encode
// list_size vectors (matches IVFBase / IVFFlat / IVFPQ getGpuVectorsEncodingSize_).
inline size_t num_warp_blocks(size_t list_size, int warp_size) {
    if (warp_size <= 0) {
        // Defensive fallback; CUDA warpSize is always 32.
        warp_size = 32;
    }
    return (list_size + static_cast<size_t>(warp_size) - 1) /
            static_cast<size_t>(warp_size);
}

// Bytes consumed by `list_size` encoded vectors, mirroring the GPU side
// implementation in IVFFlat/IVFPQ::getGpuVectorsEncodingSize_.
size_t encoded_list_bytes(
        size_t list_size,
        int bits_per_code,
        int codes_per_vec,
        bool interleaved,
        int warp_size) {
    if (interleaved) {
        // bytes to encode one code across a block of warpSize vectors
        const size_t bytes_per_dim_block =
                static_cast<size_t>(bits_per_code) *
                static_cast<size_t>(warp_size) / 8;
        const size_t bytes_per_block =
                bytes_per_dim_block * static_cast<size_t>(codes_per_vec);
        return bytes_per_block * num_warp_blocks(list_size, warp_size);
    }
    // Non-interleaved (e.g. default GpuIndexIVFPQ build): one byte per
    // sub-code per vector.
    return list_size * static_cast<size_t>(codes_per_vec);
}

size_t estimate_ivf(
        const IndexIVF* ivf,
        int device,
        const GpuClonerOptions& opts);

size_t estimate(const Index* index, int device, const GpuClonerOptions& opts);

size_t estimate_flat(const IndexFlat* flat, const GpuClonerOptions& opts) {
    // GpuIndexFlat stores either raw fp32 or fp16 vectors with no auxiliary
    // structures we need to account for.
    const size_t per_element = opts.useFloat16 ? sizeof(uint16_t) : sizeof(float);
    return static_cast<size_t>(flat->ntotal) *
            static_cast<size_t>(flat->d) * per_element;
}

size_t estimate_ivf(
        const IndexIVF* ivf,
        int device,
        const GpuClonerOptions& opts) {
    const int warp_size = faiss::gpu::getWarpSize(device);

    // Coarse quantizer: the GPU side keeps a copy of the nlist centroids.
    size_t total = static_cast<size_t>(ivf->nlist) *
            static_cast<size_t>(ivf->d) * centroid_element_bytes(opts);

    // Determine encoding parameters per concrete IVF subclass.
    int bits_per_code = 0;
    int codes_per_vec = 0;
    bool interleaved = true;
    size_t precomputed_bytes = 0;

    if (auto sq = dynamic_cast<const IndexIVFScalarQuantizer*>(ivf)) {
        // IVFScalarQuantizer + IVFFlat are stored interleaved on the GPU.
        // bits_per_code is the scalar quantizer's bit-width per dim.
        bits_per_code = static_cast<int>(sq->sq.bits);
        codes_per_vec = ivf->d;
        interleaved = true;
    } else if (auto pq = dynamic_cast<const IndexIVFPQ*>(ivf)) {
        // Default GpuIndexIVFPQ build is non-interleaved: M bytes per vector
        // when nbits == 8 (faiss only supports interleaved PQ via cuVS).
        bits_per_code = static_cast<int>(pq->pq.nbits);
        codes_per_vec = static_cast<int>(pq->pq.M);
        interleaved = false;
        if (opts.usePrecomputed) {
            const size_t ksub = static_cast<size_t>(1)
                    << static_cast<size_t>(pq->pq.nbits);
            const size_t per_elem =
                    opts.useFloat16 ? sizeof(uint16_t) : sizeof(float);
            precomputed_bytes = static_cast<size_t>(ivf->nlist) *
                    static_cast<size_t>(pq->pq.M) * ksub * per_elem;
        }
    } else if (dynamic_cast<const IndexIVFFlat*>(ivf)) {
        // Plain IVFFlat -> fp32 codes, interleaved layout.
        bits_per_code = 32;
        codes_per_vec = ivf->d;
        interleaved = true;
    } else {
        // Unknown IVF subclass: fall back to the CPU code_size as a
        // best-effort estimate (matches the existing prepareAdd fallback).
        bits_per_code = static_cast<int>(ivf->code_size) * 8;
        codes_per_vec = 1;
        interleaved = false;
    }

    total += precomputed_bytes;

    const size_t idx_bytes = indices_entry_bytes(opts.indicesOptions);

    faiss::InvertedLists* invl = ivf->invlists;
    if (invl != nullptr) {
        for (size_t i = 0; i < ivf->nlist; ++i) {
            const size_t list_size = invl->list_size(i);
            total += encoded_list_bytes(
                    list_size,
                    bits_per_code,
                    codes_per_vec,
                    interleaved,
                    warp_size);
            total += list_size * idx_bytes;
        }
    }

    // The cloner only honors reserveVecs when the source index is empty
    // (ntotal == 0); mirror that condition so the estimate stays tight.
    if (opts.reserveVecs > 0 && ivf->ntotal == 0 && ivf->nlist > 0) {
        const size_t total_reserve = static_cast<size_t>(opts.reserveVecs);
        const size_t per_list_reserve =
                (total_reserve + ivf->nlist - 1) / ivf->nlist;
        for (size_t i = 0; i < ivf->nlist; ++i) {
            total += encoded_list_bytes(
                    per_list_reserve,
                    bits_per_code,
                    codes_per_vec,
                    interleaved,
                    warp_size);
            total += per_list_reserve * idx_bytes;
        }
    }

    return total;
}

size_t estimate(const Index* index, int device, const GpuClonerOptions& opts) {
    if (index == nullptr) {
        return 0;
    }

    // Unwrap wrappers the GpuCloner forwards to Cloner::clone_Index.
    if (auto pt = dynamic_cast<const IndexPreTransform*>(index)) {
        // VectorTransforms live on CPU; the GPU footprint is dominated by
        // the wrapped sub-index.
        return estimate(pt->index, device, opts);
    }
    // IndexIDMap2 inherits from IndexIDMap, so this catches both.
    if (auto idmap = dynamic_cast<const IndexIDMap*>(index)) {
        return estimate(idmap->index, device, opts);
    }

    // GpuIndexFlat covers IndexFlat directly, and also handles an
    // IndexScalarQuantizer with QT_fp16 (it's cloned as a fp16 flat index).
    if (auto sq = dynamic_cast<const IndexScalarQuantizer*>(index)) {
        if (sq->sq.qtype == ScalarQuantizer::QT_fp16) {
            return static_cast<size_t>(sq->ntotal) *
                    static_cast<size_t>(sq->d) * sizeof(uint16_t);
        }
    }
    if (auto flat = dynamic_cast<const IndexFlat*>(index)) {
        return estimate_flat(flat, opts);
    }
    if (auto ivf = dynamic_cast<const IndexIVF*>(index)) {
        return estimate_ivf(ivf, device, opts);
    }

    // Unknown / unsupported index type. The GpuCloner will throw at clone
    // time; here we return 0 so callers can decide whether to proceed.
    return 0;
}

} // namespace

int faiss_GpuMemoryEstimate_for_cpu_index(
        const FaissIndex* index,
        int device,
        const FaissGpuClonerOptions* options,
        size_t* out) {
    try {
        if (out == nullptr) {
            return -1;
        }
        *out = 0;
        if (index == nullptr) {
            return 0;
        }
        GpuClonerOptions defaults;
        const GpuClonerOptions& opts = options != nullptr
                ? *reinterpret_cast<const GpuClonerOptions*>(options)
                : defaults;
        *out = estimate(
                reinterpret_cast<const Index*>(index), device, opts);
    }
    CATCH_AND_HANDLE
}
