// Attention with pair bias (boltzgen.model.layers.attention.AttentionPairBias) —
// the single-representation attention used in the Pairformer trunk and the
// diffusion transformer. Single example (B=1), multiplicity=1, qk_norm off
// (the inference defaults). Self-attention: k_in == s.
//
// Scalar reference for the kernel; the ggml/Vulkan version (a custom
// pair-bias attention kernel) must match this.
#pragma once

#include <vector>

namespace boltz {

struct AttnPairBiasWeights {
    int c_s = 0;     // single dim
    int c_z = 0;     // pair dim
    int num_heads = 0;
    // If false (diffusion transformer), `z` is the per-head bias directly
    // ([N*N*num_heads]) and z_norm/z_lin are unused. If true (trunk), the bias
    // is computed from the pair rep z ([N*N*c_z]) via z_norm -> z_lin.
    bool compute_pair_bias = true;
    std::vector<float> q_w, q_b;          // proj_q [c_s,c_s] + bias
    std::vector<float> k_w;               // proj_k [c_s,c_s] (no bias)
    std::vector<float> v_w;               // proj_v [c_s,c_s]
    std::vector<float> g_w;               // proj_g [c_s,c_s]
    std::vector<float> z_norm_w, z_norm_b;// LayerNorm(c_z)
    std::vector<float> z_lin_w;           // [num_heads, c_z] (no bias)
    std::vector<float> o_w;               // proj_o [c_s,c_s]
    float inf = 1e6f;
};

// s: [N*c_s], z: [N*N*c_z], mask: [N*N]. Returns [N*c_s].
std::vector<float> attention_pair_bias(const std::vector<float>& s,
                                       const std::vector<float>& z,
                                       const std::vector<float>& mask, int N,
                                       const AttnPairBiasWeights& w);

}  // namespace boltz
