// Confidence heads (boltzgen.model.modules.confidence.ConfidenceHeads, simple
// non-separate path) + the aggregated-metric (pLDDT) computation. Linear
// projections of the trunk single (s) and pair (z) representations. Single
// example (B=1).
#pragma once

#include <vector>

namespace boltz {

struct ConfidenceWeights {
    int token_s = 0, token_z = 0;
    int num_pae_bins = 0, num_pde_bins = 0, num_plddt_bins = 0;
    std::vector<float> to_pae;       // [num_pae_bins, token_z] (no bias)
    std::vector<float> to_pde;       // [num_pde_bins, token_z]
    std::vector<float> to_plddt;     // [num_plddt_bins, token_s]
    std::vector<float> to_resolved;  // [2, token_s]
};

struct ConfidenceOutputs {
    std::vector<float> pae_logits;       // [N*N*num_pae_bins]
    std::vector<float> pde_logits;       // [N*N*num_pde_bins]  (from z + z^T)
    std::vector<float> plddt_logits;     // [N*num_plddt_bins]
    std::vector<float> resolved_logits;  // [N*2]
};

// s:[N*token_s], z:[N*N*token_z].
ConfidenceOutputs confidence_heads(const std::vector<float>& s, const std::vector<float>& z,
                                   int N, const ConfidenceWeights& w);

// compute_aggregated_metric: softmax over bins, expectation over bin centers
// (centers = 0.5*bw, 1.5*bw, ... with bw = end/num_bins). `logits` is
// [rows*num_bins]; returns [rows].
std::vector<float> aggregated_metric(const std::vector<float>& logits, int rows,
                                     int num_bins, float end = 1.0f);

}  // namespace boltz
