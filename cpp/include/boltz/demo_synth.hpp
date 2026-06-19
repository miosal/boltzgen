// Shared deterministic synthetic-weight builders for the demonstration drivers
// (boltzcpp_fold, boltzcpp_design). Weights are not trained — the demos prove
// the assembled pipelines EXECUTE end to end on real structures; they are not a
// parity check (that needs the HuggingFace checkpoints, which are network-
// blocked in this environment).
#pragma once

#include "boltz/affinity.hpp"
#include "boltz/attention_pair_bias.hpp"
#include "boltz/diffusion_blocks.hpp"
#include "boltz/diffusion_transformer.hpp"
#include "boltz/trunk.hpp"

#include <cmath>
#include <vector>

namespace boltz {
namespace demo {

inline std::vector<float> fv(int n, float seed) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = 0.05f * std::sin(seed + 0.137f * i);
    return v;
}
inline std::vector<float> ones(int n) { return std::vector<float>(n, 1.0f); }
inline std::vector<float> zeros(int n) { return std::vector<float>(n, 0.0f); }

inline TriMulWeights mk_trimul(int D, float s) {
    TriMulWeights w; w.dim = D;
    w.norm_in_w = ones(D); w.norm_in_b = zeros(D); w.norm_out_w = ones(D); w.norm_out_b = zeros(D);
    w.p_in = fv(2 * D * D, s + 1); w.g_in = fv(2 * D * D, s + 2);
    w.p_out = fv(D * D, s + 3); w.g_out = fv(D * D, s + 4);
    return w;
}
inline TriAttnWeights mk_triattn(int C, int H, int hd, float s) {
    TriAttnWeights w; w.c_in = C; w.num_heads = H; w.head_dim = hd; w.inf = 1e9f;
    w.norm_w = ones(C); w.norm_b = zeros(C);
    w.tri_w = fv(H * C, s + 1); w.q_w = fv(H * hd * C, s + 2); w.k_w = fv(H * hd * C, s + 3);
    w.v_w = fv(H * hd * C, s + 4); w.g_w = fv(H * hd * C, s + 5); w.o_w = fv(C * H * hd, s + 6);
    return w;
}
inline TransitionWeights mk_trans(int dim, int hid, float s) {
    TransitionWeights t; t.dim = dim; t.hidden = hid; t.out = dim;
    t.norm_w = ones(dim); t.norm_b = zeros(dim);
    t.fc1 = fv(hid * dim, s + 1); t.fc2 = fv(hid * dim, s + 2); t.fc3 = fv(dim * hid, s + 3);
    return t;
}
inline AttnPairBiasWeights mk_attn(int cs, int cz, int H, float s) {
    AttnPairBiasWeights w; w.c_s = cs; w.c_z = cz; w.num_heads = H; w.inf = 1e6f;
    w.q_w = fv(cs * cs, s + 1); w.q_b = fv(cs, s + 1.5f); w.k_w = fv(cs * cs, s + 2);
    w.v_w = fv(cs * cs, s + 3); w.g_w = fv(cs * cs, s + 4);
    w.z_norm_w = ones(cz); w.z_norm_b = zeros(cz); w.z_lin_w = fv(H * cz, s + 5); w.o_w = fv(cs * cs, s + 6);
    return w;
}
inline PairformerWeights mk_block(int ts, int tz, int sH, int tH, int thd, float s) {
    PairformerWeights w; w.token_s = ts; w.token_z = tz;
    w.tri_mul_out = mk_trimul(tz, s + 10); w.tri_mul_in = mk_trimul(tz, s + 20);
    w.tri_att_start = mk_triattn(tz, tH, thd, s + 30); w.tri_att_end = mk_triattn(tz, tH, thd, s + 40);
    w.transition_z = mk_trans(tz, tz * 2, s + 50); w.transition_s = mk_trans(ts, ts * 2, s + 60);
    w.pre_norm_s_w = ones(ts); w.pre_norm_s_b = zeros(ts);
    w.attention = mk_attn(ts, tz, sH, s + 70);
    return w;
}
inline AdaLNWeights mk_adaln(int dim, int dc, float s) {
    AdaLNWeights w; w.dim = dim; w.dim_cond = dc;
    w.s_norm_w = ones(dc); w.s_scale_w = fv(dim * dc, s + 1); w.s_scale_b = fv(dim, s + 2); w.s_bias_w = fv(dim * dc, s + 3);
    return w;
}
inline DiffusionTransformerLayerWeights mk_difflayer(int dim, int dc, int H, float s) {
    DiffusionTransformerLayerWeights w; w.dim = dim; w.dim_cond = dc; w.num_heads = H;
    w.adaln = mk_adaln(dim, dc, s);
    w.attn.c_s = dim; w.attn.c_z = 0; w.attn.num_heads = H; w.attn.compute_pair_bias = false; w.attn.inf = 1e6f;
    w.attn.q_w = fv(dim * dim, s + 10); w.attn.q_b = fv(dim, s + 10.5f); w.attn.k_w = fv(dim * dim, s + 11);
    w.attn.v_w = fv(dim * dim, s + 12); w.attn.g_w = fv(dim * dim, s + 13); w.attn.o_w = fv(dim * dim, s + 14);
    w.op_w = fv(dim * dc, s + 15); w.op_b = fv(dim, s + 16);
    w.transition.dim = dim; w.transition.dim_cond = dc; w.transition.dim_inner = dim * 2;
    w.transition.adaln = mk_adaln(dim, dc, s + 20);
    w.transition.swish_gate_w = fv(2 * (dim * 2) * dim, s + 21);
    w.transition.a_to_b_w = fv((dim * 2) * dim, s + 22);
    w.transition.b_to_a_w = fv(dim * (dim * 2), s + 23);
    w.transition.op_w = fv(dim * dc, s + 24); w.transition.op_b = fv(dim, s + 25);
    return w;
}

}  // namespace demo
}  // namespace boltz
