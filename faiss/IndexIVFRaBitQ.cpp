/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFRaBitQ.h>

#include <omp.h>

#include <faiss/OMPConfig.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/utils/rabitq_simd.h>

namespace faiss {

IndexIVFRaBitQ::IndexIVFRaBitQ(
        Index* quantizer,
        const size_t d,
        const size_t nlist,
        MetricType metric,
        bool own_invlists,
        uint8_t nb_bits_in)
        : IndexIVF(quantizer, d, nlist, 0, metric, own_invlists),
          rabitq(d, metric, nb_bits_in) {
    code_size = rabitq.code_size;
    if (own_invlists) {
        invlists->code_size = code_size;
    }
    is_trained = false;

    by_residual = true;
}

IndexIVFRaBitQ::IndexIVFRaBitQ() {
    by_residual = true;
}

void IndexIVFRaBitQ::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* assign) {
    rabitq.train(n, x);
}

void IndexIVFRaBitQ::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    memset(codes, 0, (code_size + coarse_size) * n);

#pragma omp parallel if (n > 1000) num_threads(num_omp_threads)
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            if (list_no >= 0) {
                const float* xi = x + i * d;
                uint8_t* code = codes + i * (code_size + coarse_size);

                // both by_residual and !by_residual lead to the same code
                quantizer->reconstruct(list_no, centroid.data());
                rabitq.compute_codes_core(
                        xi, code + coarse_size, 1, centroid.data());

                if (coarse_size) {
                    encode_listno(list_no, code);
                }
            }
        }
    }
}

void IndexIVFRaBitQ::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* listnos,
        float* x) const {
#pragma omp parallel num_threads(num_omp_threads)
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * code_size;
            int64_t list_no = listnos[i];
            float* xi = x + i * d;

            quantizer->reconstruct(list_no, centroid.data());
            rabitq.decode_core(code, xi, 1, centroid.data());
        }
    }
}

void IndexIVFRaBitQ::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);

    DirectMapAdd dm_add(direct_map, n, xids);

#pragma omp parallel num_threads(num_omp_threads)
    {
        std::vector<uint8_t> one_code(code_size);
        std::vector<float> centroid(d);

        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = precomputed_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                int64_t id = xids ? xids[i] : ntotal + i;

                const float* xi = x + i * d;

                // both by_residual and !by_residual lead to the same code
                quantizer->reconstruct(list_no, centroid.data());
                rabitq.compute_codes_core(
                        xi, one_code.data(), 1, centroid.data());

                size_t ofs = invlists->add_entry(
                        list_no, id, one_code.data(), inverted_list_context);

                dm_add.add(i, list_no, ofs);

            } else if (rank == 0 && list_no == -1) {
                dm_add.add(i, -1, 0);
            }
        }
    }

    ntotal += n;
}

struct RaBitInvertedListScanner : InvertedListScanner {
    const IndexIVFRaBitQ& ivf_rabitq;

    std::vector<float> reconstructed_centroid;
    std::vector<float> query_vector;

    std::unique_ptr<FlatCodesDistanceComputer> dc;
    RaBitQDistanceComputer* rabitq_dc =
            nullptr; // For multi-bit adaptive filtering

    uint8_t qb = 0;
    bool centered = false;

    explicit RaBitInvertedListScanner(
            const IndexIVFRaBitQ& ivf_rabitq_in,
            bool store_pairs = false,
            const IDSelector* sel = nullptr,
            uint8_t qb_in = 0,
            bool centered = false)
            : InvertedListScanner(store_pairs, sel),
              ivf_rabitq{ivf_rabitq_in},
              qb{qb_in},
              centered(centered) {
        keep_max = is_similarity_metric(ivf_rabitq.metric_type);
        code_size = ivf_rabitq.code_size;
    }

    /// from now on we handle this query.
    void set_query(const float* query_vector_in) override {
        query_vector.assign(query_vector_in, query_vector_in + ivf_rabitq.d);

        internal_try_setup_dc();
    }

    /// following codes come from this inverted list
    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;

        reconstructed_centroid.resize(ivf_rabitq.d);
        ivf_rabitq.quantizer->reconstruct(
                list_no, reconstructed_centroid.data());

        internal_try_setup_dc();
    }

    /// compute a single query-to-code distance
    float distance_to_code(const uint8_t* code) const override {
        return dc->distance_to_code(code);
    }

    /// Override scan_codes to implement adaptive filtering for multi-bit codes
    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t ex_bits = ivf_rabitq.rabitq.nb_bits - 1;

        // For 1-bit codes, use default implementation
        if (ex_bits == 0 || rabitq_dc == nullptr) {
            return InvertedListScanner::scan_codes(
                    list_size, codes, ids, simi, idxi, k);
        }

        // Multi-bit: Two-stage search with adaptive filtering
        size_t nup = 0;

        // Stats tracking for multi-bit two-stage search
        // n_1bit_evaluations: candidates evaluated using 1-bit lower bound
        // n_multibit_evaluations: candidates requiring full multi-bit distance
        size_t local_1bit_evaluations = 0;
        size_t local_multibit_evaluations = 0;

        for (size_t j = 0; j < list_size; j++) {
            if (sel != nullptr) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                if (!sel->is_member(id)) {
                    codes += code_size;
                    continue;
                }
            }

            local_1bit_evaluations++;

            // Stage 1: Compute lower bound using 1-bit codes
            float lower_bound = rabitq_dc->lower_bound_distance(codes);

            // Stage 2: Adaptive filtering
            // L2 (min-heap): filter if lower_bound < simi[0]
            // IP (max-heap): filter if lower_bound > simi[0]
            // Note: Using simi[0] directly (not cached) enables more aggressive
            // filtering as the heap is updated with better candidates
            bool should_refine = keep_max ? (lower_bound > simi[0])
                                          : (lower_bound < simi[0]);

            if (should_refine) {
                local_multibit_evaluations++;
                // Lower bound is promising, compute full distance
                float dis = distance_to_code(codes);

                // Check if distance improves heap
                bool improves_heap =
                        keep_max ? (dis > simi[0]) : (dis < simi[0]);

                if (improves_heap) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    if (keep_max) {
                        minheap_replace_top(k, simi, idxi, dis, id);
                    } else {
                        maxheap_replace_top(k, simi, idxi, dis, id);
                    }
                    nup++;
                }
            }
            codes += code_size;
        }

        // Update global stats atomically
#pragma omp atomic
        rabitq_stats.n_1bit_evaluations += local_1bit_evaluations;
#pragma omp atomic
        rabitq_stats.n_multibit_evaluations += local_multibit_evaluations;

        return nup;
    }

    void internal_try_setup_dc() {
        if (!query_vector.empty() && !reconstructed_centroid.empty()) {
            // both query_vector and centroid are available!
            // set up DistanceComputer
            dc.reset(ivf_rabitq.rabitq.get_distance_computer(
                    qb, reconstructed_centroid.data(), centered));

            dc->set_query(query_vector.data());

            // Try to cast to RaBitQDistanceComputer for multi-bit support
            rabitq_dc = dynamic_cast<RaBitQDistanceComputer*>(dc.get());
        }
    }
};

InvertedListScanner* IndexIVFRaBitQ::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters* search_params_in) const {
    uint8_t used_qb = qb;
    bool centered = false;
    if (auto params = dynamic_cast<const IVFRaBitQSearchParameters*>(
                search_params_in)) {
        used_qb = params->qb;
        centered = params->centered;
    }

    return new RaBitInvertedListScanner(
            *this, store_pairs, sel, used_qb, centered);
}

void IndexIVFRaBitQ::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);

    std::vector<float> centroid(d);
    quantizer->reconstruct(list_no, centroid.data());

    rabitq.decode_core(code, recons, 1, centroid.data());
}

void IndexIVFRaBitQ::sa_decode(idx_t n, const uint8_t* codes, float* x) const {
    size_t coarse_size = coarse_code_size();

#pragma omp parallel num_threads(num_omp_threads)
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * (code_size + coarse_size);
            int64_t list_no = decode_listno(code);
            float* xi = x + i * d;

            quantizer->reconstruct(list_no, centroid.data());
            rabitq.decode_core(code + coarse_size, xi, 1, centroid.data());
        }
    }
}

void IndexIVFRaBitQ::compute_distance_to_codes_for_list(
        const idx_t list_no,
        const float* x,
        idx_t n,
        const uint8_t* codes,
        float* dists,
        float* /*dist_table*/) const {
    FAISS_THROW_IF_NOT(n >= 0);
    FAISS_THROW_IF_NOT(list_no >= 0 && (size_t)list_no < nlist);
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(codes != nullptr);
    FAISS_THROW_IF_NOT(dists != nullptr);
    FAISS_THROW_IF_NOT(code_size > 0);
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(
            (size_t)n <= (std::numeric_limits<size_t>::max)() / code_size);

    // RaBitQ uses per-vector correction factors stored in the codes, so we
    // must use the RaBitQuantizer distance computer.
    std::vector<float> centroid(d);
    quantizer->reconstruct(list_no, centroid.data());

    // NOTE: centered=false is hardcoded here and in
    // compute_distance_to_codes_with_precomputed. Do not flip without also
    // updating the inline distance loop in the precomputed path.
    std::unique_ptr<FlatCodesDistanceComputer> dc(
            rabitq.get_distance_computer(
                    qb, centroid.data(), /*centered=*/false));
    dc->set_query(x);

    const uint8_t* code = codes;
    for (idx_t i = 0; i < n; i++, code += code_size) {
        dists[i] = dc->distance_to_code(code);
    }
}

size_t IndexIVFRaBitQ::query_bitplanes_size() const {
    FAISS_THROW_IF_NOT(qb > 0 && qb <= 8);
    return sizeof(PrecomputedQueryHeader) + rabitq.precomputed_query_size(qb);
}

void IndexIVFRaBitQ::compute_distance_to_codes_with_precomputed(
        idx_t list_no,
        const float* x,
        idx_t n,
        const uint8_t* codes,
        float* dists,
        uint8_t* query_bp,
        size_t* query_bp_size) const {
    FAISS_THROW_IF_NOT(qb > 0 && qb <= 8);
    FAISS_THROW_IF_NOT(n >= 0);
    FAISS_THROW_IF_NOT(list_no >= 0 && (size_t)list_no < nlist);
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(codes != nullptr);
    FAISS_THROW_IF_NOT(dists != nullptr);
    FAISS_THROW_IF_NOT(code_size > 0);
    FAISS_THROW_IF_NOT(query_bp != nullptr);
    FAISS_THROW_IF_NOT(query_bp_size != nullptr);
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(
            (size_t)n <= (std::numeric_limits<size_t>::max)() / code_size);

    size_t header_size = sizeof(PrecomputedQueryHeader);
    size_t byte_dim = (d + 7) / 8;
    size_t required_bp_size = query_bitplanes_size();

    // A buffer is reusable only if:
    //   (1) it has the size we expect, AND
    //   (2) its header identifies the same (list_no, qb, nb_bits, d, magic,
    //       version) we're being called with.
    // Either mismatch triggers a fresh compute. This catches every silent
    // cross-misuse class: wrong list, wrong index, wrong qb, stale layout.
    bool use_precomputed = false;
    if (*query_bp_size == required_bp_size) {
        const auto* hdr =
                reinterpret_cast<const PrecomputedQueryHeader*>(query_bp);
        use_precomputed =
                hdr->magic == PrecomputedQueryHeader::kMagic &&
                hdr->version == PrecomputedQueryHeader::kVersion &&
                hdr->nb_bits == rabitq.nb_bits &&
                hdr->qb == qb &&
                hdr->d == (uint32_t)d &&
                hdr->list_no == list_no;
    }

    if (!use_precomputed) {
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        // Stamp the header first. compute_query_precomputed writes after it.
        auto* hdr = reinterpret_cast<PrecomputedQueryHeader*>(query_bp);
        hdr->magic = PrecomputedQueryHeader::kMagic;
        hdr->version = PrecomputedQueryHeader::kVersion;
        hdr->nb_bits = (uint16_t)rabitq.nb_bits;
        hdr->qb = qb;
        hdr->pad[0] = hdr->pad[1] = hdr->pad[2] = 0;
        hdr->d = (uint32_t)d;
        hdr->list_no = list_no;

        size_t written = 0;
        // NOTE: centered=false is hardcoded here and in
        // compute_distance_to_codes_for_list. Do not flip without also
        // updating the inline distance loop below (which only implements the
        // !centered branch).
        rabitq.compute_query_precomputed(
                x, qb, centroid.data(), /*centered=*/false,
                query_bp + header_size, &written);

        FAISS_THROW_IF_NOT(written == rabitq.precomputed_query_size(qb));
        *query_bp_size = header_size + written;
    }
    FAISS_THROW_IF_NOT(*query_bp_size == required_bp_size);

    // Zero-copy: read directly from the precomputed buffer past the header.
    size_t offset = header_size;

    const auto* query_fac = reinterpret_cast<
            const PrecomputedQueryScalars*>(query_bp + offset);
    offset += sizeof(PrecomputedQueryScalars);

    const float* rotated_q =
            reinterpret_cast<const float*>(query_bp + offset);
    offset += d * sizeof(float);

    const uint8_t* rearranged_qq = query_bp + offset;

    // Compute distances inline rather than via get_distance_computer +
    // set_query + distance_to_code: get_distance_computer alone costs
    // ~1.5 us per call, which dominates 1-bit branch at the small batch
    // sizes typical in certain workloads.
    //
    // - Multi-bit branch delegates to
    //   rabitq_utils::compute_full_multibit_distance (same utility used
    //   by RaBitQDistanceComputerQ::distance_to_code_full), so changes
    //   there propagate automatically.
    // - 1-bit branch reproduces the math from
    //   RaBitQDistanceComputerQ::distance_to_code_1bit. Keep in sync.
    const size_t ex_bits = rabitq.nb_bits - 1;
    const uint8_t* code = codes;
    for (idx_t i = 0; i < n; i++, code += code_size) {
        if (ex_bits == 0) {
            // 1-bit path (same as distance_to_code_1bit)
            const uint8_t* binary_data = code;
            const auto* base_fac = reinterpret_cast<
                    const rabitq_utils::SignBitFactors*>(
                    code + byte_dim);

            // SIMD dot products on rearranged bitplanes
            auto dot_qo = rabitq::bitwise_and_dot_product(
                    rearranged_qq, binary_data, byte_dim, qb);
            auto sum_q = rabitq::popcount(binary_data, byte_dim);

            // Match exact operation sequence from
            // RaBitQDistanceComputerQ::distance_to_code_1bit
            float final_dot = 0;
            final_dot += query_fac->c1 * dot_qo;
            final_dot += query_fac->c2 * sum_q;
            final_dot -= query_fac->c34;

            float pre_dist = base_fac->or_minus_c_l2sqr +
                    query_fac->qr_to_c_L2sqr -
                    2 * base_fac->dp_multiplier * final_dot;

            if (metric_type == MetricType::METRIC_L2) {
                dists[i] = pre_dist;
            } else {
                dists[i] = -0.5f * (pre_dist - query_fac->qr_norm_L2sqr);
            }
        } else {
            // Multi-bit path (same as distance_to_code_full)
            const uint8_t* binary_data = code;
            size_t fac_offset =
                    byte_dim + sizeof(rabitq_utils::SignBitFactorsWithError);
            const uint8_t* ex_code = code + fac_offset;
            const auto* ex_fac = reinterpret_cast<
                    const rabitq_utils::ExtraBitsFactors*>(
                    ex_code + (d * ex_bits + 7) / 8);

            dists[i] = rabitq_utils::compute_full_multibit_distance(
                    binary_data,
                    ex_code,
                    *ex_fac,
                    rotated_q,
                    query_fac->qr_to_c_L2sqr,
                    query_fac->qr_norm_L2sqr,
                    d,
                    ex_bits,
                    metric_type);
        }
    }
}

struct IVFRaBitDistanceComputer : DistanceComputer {
    const float* q = nullptr;
    const IndexIVFRaBitQ* parent = nullptr;

    void set_query(const float* x) override;

    float operator()(idx_t i) override;

    float symmetric_dis(idx_t i, idx_t j) override;
};

void IVFRaBitDistanceComputer::set_query(const float* x) {
    q = x;
}

float IVFRaBitDistanceComputer::operator()(idx_t i) {
    // find the appropriate list
    idx_t lo = parent->direct_map.get(i);
    uint64_t list_no = lo_listno(lo);
    uint64_t offset = lo_offset(lo);

    const uint8_t* code = parent->invlists->get_single_code(list_no, offset);

    // ok, we know the appropriate cluster that we need
    std::vector<float> centroid(parent->d);
    parent->quantizer->reconstruct(list_no, centroid.data());

    // compute the distance
    float distance = 0;

    std::unique_ptr<FlatCodesDistanceComputer> dc(
            parent->rabitq.get_distance_computer(
                    parent->qb, centroid.data(), /*centered=*/false));
    dc->set_query(q);
    distance = dc->distance_to_code(code);

    // deallocate
    parent->invlists->release_codes(list_no, code);

    // done
    return distance;
}

float IVFRaBitDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    FAISS_THROW_MSG("Not implemented");
}

DistanceComputer* IndexIVFRaBitQ::get_distance_computer() const {
    IVFRaBitDistanceComputer* dc = new IVFRaBitDistanceComputer;
    dc->parent = this;
    return dc;
}

} // namespace faiss
