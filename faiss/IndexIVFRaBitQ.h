/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/Index.h>
#include <faiss/IndexIVF.h>

#include <faiss/impl/RaBitQStats.h>
#include <faiss/impl/RaBitQuantizer.h>

namespace faiss {

struct IVFRaBitQSearchParameters : IVFSearchParameters {
    uint8_t qb = 0;
    bool centered = false;
};

// * by_residual is true, just by design
struct IndexIVFRaBitQ : IndexIVF {
    RaBitQuantizer rabitq;

    // the default number of bits to quantize a query with.
    // use '0' to disable quantization and use raw fp32 values.
    uint8_t qb = 0;

    IndexIVFRaBitQ(
            Index* quantizer,
            const size_t d,
            const size_t nlist,
            MetricType metric = METRIC_L2,
            bool own_invlists = true,
            uint8_t nb_bits = 1);

    IndexIVFRaBitQ();

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void decode_vectors(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            float* x) const override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void compute_distance_to_codes_for_list(
            const idx_t list_no,
            const float* x,
            idx_t n,
            const uint8_t* codes,
            float* dists,
            float* dist_table) const override;

    /// Self-describing header stamped at the start of the precomputed query
    /// buffer. Validated on every reuse so a buffer built for a different
    /// list / qb / nb_bits / d fails loud instead of silently producing
    /// wrong distances. POD, fixed 24 bytes.
    struct PrecomputedQueryHeader {
        static constexpr uint32_t kMagic = 0x52424351; // 'RBCQ'
        static constexpr uint16_t kVersion = 1; // bump on layout change

        uint32_t magic;
        uint16_t version;
        uint16_t nb_bits;
        uint8_t qb;
        uint8_t pad[3];
        uint32_t d;
        int64_t list_no;
    };
    static_assert(
            sizeof(PrecomputedQueryHeader) == 24,
            "PrecomputedQueryHeader must be exactly 24 bytes");

    /// Compute distances to codes with optional precomputed query state.
    ///
    /// The buffer carries a self-describing header so the function can
    /// distinguish a fresh reusable buffer from a stale / wrong-list /
    /// wrong-config one without trusting the caller's bookkeeping. The
    /// caller MUST allocate at least query_bitplanes_size() bytes.
    ///
    /// @param list_no       IVF list the codes belong to.
    /// @param x             query vector (d floats).
    /// @param n             number of codes.
    /// @param codes         input codes, n * code_size bytes.
    /// @param dists         output distances, n floats.
    /// @param query_bp      caller-allocated buffer of query_bitplanes_size()
    ///                      bytes.
    /// @param query_bp_size in/out: 0 to (re)compute, otherwise must equal
    ///                      query_bitplanes_size() AND the in-buffer header
    ///                      must match (list_no, qb, nb_bits, d). Either
    ///                      mismatch triggers a fresh compute. On return:
    ///                      set to bytes written.
    void compute_distance_to_codes_with_precomputed(
            idx_t list_no,
            const float* x,
            idx_t n,
            const uint8_t* codes,
            float* dists,
            uint8_t* query_bp,
            size_t* query_bp_size) const;

    /// Returns the byte size needed for precomputed query bitplanes buffer.
    /// Includes the self-describing header.
    size_t query_bitplanes_size() const;

    // unfortunately
    DistanceComputer* get_distance_computer() const override;
};

} // namespace faiss
