// Windowed atom attention (boltzgen.model.modules.transformers.AtomTransformer,
// AF3 Algorithm 7 core): the atom sequence is split into query windows of W
// atoms; each window cross-attends to a block of H keys gathered from the
// surrounding half-windows (single_to_keys). Boundary-filled keys are masked.
// Single example (B=1).
#pragma once

#include "boltz/attention_pair_bias.hpp"

#include <vector>

namespace boltz {

// q: [N*d] queries, kv: [N*d] key/value source (N = K*W). per_head_bias may be
// empty (treated as zeros) or [K*W*H*num_heads] (per window). attn must have
// compute_pair_bias=false. Returns [N*d].
std::vector<float> windowed_atom_attention(const std::vector<float>& q,
                                           const std::vector<float>& kv, int N, int d, int W,
                                           int H, const AttnPairBiasWeights& attn,
                                           const std::vector<float>& per_head_bias = {});

}  // namespace boltz
