// Pairformer layer (boltzgen.model.layers.pairformer.PairformerLayer) — the
// repeating trunk block of the folding / design / affinity models. Assembles the
// validated pair ops + single-rep attention with the reference residual order
// (inference: dropout is a no-op, post-LayerNorm off). Single example (B=1).
//
//   z += tri_mul_out(z); z += tri_mul_in(z);
//   z += tri_att_start(z); z += tri_att_end(z); z += transition_z(z)
//   s += attention(pre_norm(s), z); s += transition_s(s)
#pragma once

#include "boltz/attention_pair_bias.hpp"
#include "boltz/triangle_attention.hpp"
#include "boltz/triangle_mult.hpp"

#include <vector>

namespace boltz {

// SwiGLU transition weights: norm -> silu(fc1)*fc2 -> fc3 (all Linears bias-free).
struct TransitionWeights {
    int dim = 0, hidden = 0, out = 0;
    std::vector<float> norm_w, norm_b;  // [dim]
    std::vector<float> fc1, fc2;        // [hidden, dim]
    std::vector<float> fc3;             // [out, hidden]
};

struct PairformerWeights {
    int token_s = 0, token_z = 0;
    TriMulWeights tri_mul_out, tri_mul_in;
    TriAttnWeights tri_att_start, tri_att_end;
    TransitionWeights transition_z, transition_s;
    std::vector<float> pre_norm_s_w, pre_norm_s_b;  // LayerNorm(token_s)
    AttnPairBiasWeights attention;
};

struct PairformerState {
    std::vector<float> s;  // [N*token_s]
    std::vector<float> z;  // [N*N*token_z]
};

// token_mask: [N] (1=valid). pair_mask: [N*N]. Returns updated s, z.
PairformerState pairformer_block(const std::vector<float>& s, const std::vector<float>& z,
                                 const std::vector<float>& token_mask,
                                 const std::vector<float>& pair_mask, int N,
                                 const PairformerWeights& w);

}  // namespace boltz
