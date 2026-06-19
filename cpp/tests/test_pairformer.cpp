// Wiring test for the assembled Pairformer block: compare the block to an
// independent composition of the (already equivalence-tested) public ops, to
// verify residual order, that updates feed forward (tri_mul_in sees updated z,
// attention sees post-pair-stack z), and mask plumbing.
#include "boltz/pairformer.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

void fill(std::vector<float>& v, int n, float seed) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.05f * std::sin(seed + 0.31f * i);
}

TriMulWeights make_trimul(int D, float seed) {
    TriMulWeights w; w.dim = D;
    w.norm_in_w.assign(D, 1.0f); w.norm_in_b.assign(D, 0.0f);
    w.norm_out_w.assign(D, 1.0f); w.norm_out_b.assign(D, 0.0f);
    fill(w.p_in, 2 * D * D, seed + 1); fill(w.g_in, 2 * D * D, seed + 2);
    fill(w.p_out, D * D, seed + 3); fill(w.g_out, D * D, seed + 4);
    return w;
}

TriAttnWeights make_triattn(int C, int H, int hd, float seed) {
    TriAttnWeights w; w.c_in = C; w.num_heads = H; w.head_dim = hd; w.inf = 1e9f;
    w.norm_w.assign(C, 1.0f); w.norm_b.assign(C, 0.0f);
    fill(w.tri_w, H * C, seed + 1); fill(w.q_w, H * hd * C, seed + 2);
    fill(w.k_w, H * hd * C, seed + 3); fill(w.v_w, H * hd * C, seed + 4);
    fill(w.g_w, H * hd * C, seed + 5); fill(w.o_w, C * H * hd, seed + 6);
    return w;
}

TransitionWeights make_trans(int dim, int hidden, float seed) {
    TransitionWeights t; t.dim = dim; t.hidden = hidden; t.out = dim;
    t.norm_w.assign(dim, 1.0f); t.norm_b.assign(dim, 0.0f);
    fill(t.fc1, hidden * dim, seed + 1); fill(t.fc2, hidden * dim, seed + 2);
    fill(t.fc3, dim * hidden, seed + 3);
    return t;
}

AttnPairBiasWeights make_attn(int c_s, int c_z, int H, float seed) {
    AttnPairBiasWeights w; w.c_s = c_s; w.c_z = c_z; w.num_heads = H; w.inf = 1e6f;
    fill(w.q_w, c_s * c_s, seed + 1); fill(w.q_b, c_s, seed + 1.5f);
    fill(w.k_w, c_s * c_s, seed + 2); fill(w.v_w, c_s * c_s, seed + 3);
    fill(w.g_w, c_s * c_s, seed + 4);
    w.z_norm_w.assign(c_z, 1.0f); w.z_norm_b.assign(c_z, 0.0f);
    fill(w.z_lin_w, H * c_z, seed + 5); fill(w.o_w, c_s * c_s, seed + 6);
    return w;
}

void ln(const float* x, int D, float* out) {
    float mu = 0; for (int d = 0; d < D; ++d) mu += x[d]; mu /= D;
    float var = 0; for (int d = 0; d < D; ++d) var += (x[d] - mu) * (x[d] - mu); var /= D;
    float iv = 1.0f / std::sqrt(var + 1e-5f);
    for (int d = 0; d < D; ++d) out[d] = (x[d] - mu) * iv;  // unit affine
}

std::vector<float> trans_apply(const std::vector<float>& x, int rows, const TransitionWeights& t) {
    std::vector<float> out((size_t)rows * t.out);
    auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
    for (int r = 0; r < rows; ++r) {
        std::vector<float> n(t.dim); ln(&x[r * t.dim], t.dim, n.data());
        std::vector<float> h(t.hidden);
        for (int o = 0; o < t.hidden; ++o) {
            float a = 0, b = 0;
            for (int i = 0; i < t.dim; ++i) { a += t.fc1[o * t.dim + i] * n[i]; b += t.fc2[o * t.dim + i] * n[i]; }
            h[o] = silu(a) * b;
        }
        for (int o = 0; o < t.out; ++o) { float s = 0; for (int i = 0; i < t.hidden; ++i) s += t.fc3[o * t.hidden + i] * h[i]; out[r * t.out + o] = s; }
    }
    return out;
}

}  // namespace

BOLTZ_TEST(pairformer_block_matches_composition) {
    const int N = 3, ts = 4, tz = 4, sH = 2, tH = 2, thd = 2, hid = 8;
    PairformerWeights w;
    w.token_s = ts; w.token_z = tz;
    w.tri_mul_out = make_trimul(tz, 10);
    w.tri_mul_in = make_trimul(tz, 20);
    w.tri_att_start = make_triattn(tz, tH, thd, 30);
    w.tri_att_end = make_triattn(tz, tH, thd, 40);
    w.transition_z = make_trans(tz, hid, 50);
    w.transition_s = make_trans(ts, hid, 60);
    w.pre_norm_s_w.assign(ts, 1.0f); w.pre_norm_s_b.assign(ts, 0.0f);
    w.attention = make_attn(ts, tz, sH, 70);

    std::vector<float> s(N * ts), z(N * N * tz), tok_mask(N, 1.0f), pair_mask(N * N, 1.0f);
    for (int i = 0; i < N * ts; ++i) s[i] = 0.2f * std::sin(0.3f + i);
    for (int i = 0; i < N * N * tz; ++i) z[i] = 0.15f * std::cos(0.2f + i);
    tok_mask[2] = 0.0f;
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) pair_mask[i * N + j] = (i != 2 && j != 2) ? 1.0f : 0.0f;

    auto got = pairformer_block(s, z, tok_mask, pair_mask, N, w);

    // Independent composition.
    std::vector<float> rz = z;
    auto addv = [](std::vector<float>& d, const std::vector<float>& s) { for (size_t i = 0; i < d.size(); ++i) d[i] += s[i]; };
    addv(rz, triangle_multiplication(rz, pair_mask, N, tz, w.tri_mul_out, false));
    addv(rz, triangle_multiplication(rz, pair_mask, N, tz, w.tri_mul_in, true));
    addv(rz, triangle_attention(rz, pair_mask, N, w.tri_att_start, true));
    addv(rz, triangle_attention(rz, pair_mask, N, w.tri_att_end, false));
    addv(rz, trans_apply(rz, N * N, w.transition_z));

    std::vector<float> sn(N * ts);
    for (int i = 0; i < N; ++i) ln(&s[i * ts], ts, &sn[i * ts]);
    std::vector<float> mask2d(N * N);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) mask2d[i * N + j] = tok_mask[j];
    std::vector<float> rs = s;
    addv(rs, attention_pair_bias(sn, rz, mask2d, N, w.attention));
    addv(rs, trans_apply(rs, N, w.transition_s));

    expect_eq_i(static_cast<long>(got.s.size()), N * ts, "s shape");
    expect_eq_i(static_cast<long>(got.z.size()), N * N * tz, "z shape");
    for (size_t i = 0; i < rs.size(); ++i)
        if (std::fabs(got.s[i] - rs[i]) > 1e-4f) throw AssertFailure("s mismatch " + std::to_string(i));
    for (size_t i = 0; i < rz.size(); ++i)
        if (std::fabs(got.z[i] - rz[i]) > 1e-4f) throw AssertFailure("z mismatch " + std::to_string(i));
    // Sanity: the block actually changed the inputs.
    bool changed = false;
    for (size_t i = 0; i < rs.size(); ++i) if (std::fabs(got.s[i] - s[i]) > 1e-6f) { changed = true; break; }
    expect_true(changed, "block updates s");
}

int main() {
    std::printf("== test_pairformer ==\n");
    return boltztest::run_all();
}
