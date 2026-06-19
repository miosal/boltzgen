// Tests for the Pairformer trunk (block stack) and the distogram head.
#include "boltz/trunk.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

void fill(std::vector<float>& v, int n, float seed) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.05f * std::sin(seed + 0.29f * i);
}
TriMulWeights tm(int D, float s) {
    TriMulWeights w; w.dim = D;
    w.norm_in_w.assign(D, 1); w.norm_in_b.assign(D, 0);
    w.norm_out_w.assign(D, 1); w.norm_out_b.assign(D, 0);
    fill(w.p_in, 2 * D * D, s + 1); fill(w.g_in, 2 * D * D, s + 2);
    fill(w.p_out, D * D, s + 3); fill(w.g_out, D * D, s + 4);
    return w;
}
TriAttnWeights ta(int C, int H, int hd, float s) {
    TriAttnWeights w; w.c_in = C; w.num_heads = H; w.head_dim = hd; w.inf = 1e9f;
    w.norm_w.assign(C, 1); w.norm_b.assign(C, 0);
    fill(w.tri_w, H * C, s + 1); fill(w.q_w, H * hd * C, s + 2); fill(w.k_w, H * hd * C, s + 3);
    fill(w.v_w, H * hd * C, s + 4); fill(w.g_w, H * hd * C, s + 5); fill(w.o_w, C * H * hd, s + 6);
    return w;
}
TransitionWeights tr(int dim, int hid, float s) {
    TransitionWeights t; t.dim = dim; t.hidden = hid; t.out = dim;
    t.norm_w.assign(dim, 1); t.norm_b.assign(dim, 0);
    fill(t.fc1, hid * dim, s + 1); fill(t.fc2, hid * dim, s + 2); fill(t.fc3, dim * hid, s + 3);
    return t;
}
AttnPairBiasWeights ab(int cs, int cz, int H, float s) {
    AttnPairBiasWeights w; w.c_s = cs; w.c_z = cz; w.num_heads = H; w.inf = 1e6f;
    fill(w.q_w, cs * cs, s + 1); fill(w.q_b, cs, s + 1.5f); fill(w.k_w, cs * cs, s + 2);
    fill(w.v_w, cs * cs, s + 3); fill(w.g_w, cs * cs, s + 4);
    w.z_norm_w.assign(cz, 1); w.z_norm_b.assign(cz, 0); fill(w.z_lin_w, H * cz, s + 5); fill(w.o_w, cs * cs, s + 6);
    return w;
}
PairformerWeights block(int ts, int tz, float s) {
    PairformerWeights w; w.token_s = ts; w.token_z = tz;
    w.tri_mul_out = tm(tz, s + 10); w.tri_mul_in = tm(tz, s + 20);
    w.tri_att_start = ta(tz, 2, 2, s + 30); w.tri_att_end = ta(tz, 2, 2, s + 40);
    w.transition_z = tr(tz, 8, s + 50); w.transition_s = tr(ts, 8, s + 60);
    w.pre_norm_s_w.assign(ts, 1); w.pre_norm_s_b.assign(ts, 0);
    w.attention = ab(ts, tz, 2, s + 70);
    return w;
}

}  // namespace

BOLTZ_TEST(trunk_equals_sequential_blocks) {
    const int N = 3, ts = 4, tz = 4;
    std::vector<PairformerWeights> blocks = {block(ts, tz, 0), block(ts, tz, 100)};
    std::vector<float> s(N * ts), z(N * N * tz), tmask(N, 1.0f), pmask(N * N, 1.0f);
    for (int i = 0; i < N * ts; ++i) s[i] = 0.2f * std::sin(0.3f + i);
    for (int i = 0; i < N * N * tz; ++i) z[i] = 0.15f * std::cos(0.2f + i);

    auto trunk = pairformer_trunk(s, z, tmask, pmask, N, blocks);

    auto b0 = pairformer_block(s, z, tmask, pmask, N, blocks[0]);
    auto b1 = pairformer_block(b0.s, b0.z, tmask, pmask, N, blocks[1]);

    for (size_t i = 0; i < b1.s.size(); ++i)
        if (std::fabs(trunk.s[i] - b1.s[i]) > 1e-5f) throw AssertFailure("trunk s mismatch");
    for (size_t i = 0; i < b1.z.size(); ++i)
        if (std::fabs(trunk.z[i] - b1.z[i]) > 1e-5f) throw AssertFailure("trunk z mismatch");
}

BOLTZ_TEST(distogram_symmetric_and_correct) {
    const int N = 3, tz = 4, bins = 5;
    std::vector<float> z(N * N * tz);
    for (int i = 0; i < N * N * tz; ++i) z[i] = 0.1f * std::sin(0.5f + i);
    std::vector<float> lw(bins * tz), lb(bins);
    for (int i = 0; i < bins * tz; ++i) lw[i] = 0.07f * std::cos(i);
    for (int i = 0; i < bins; ++i) lb[i] = 0.01f * i;

    auto d = distogram_head(z, N, tz, bins, lw, lb);
    expect_eq_i(static_cast<long>(d.size()), N * N * bins, "shape");

    // Output is symmetric in (i,j) because z was symmetrized.
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int b = 0; b < bins; ++b)
                expect_eq_f(d[(i * N + j) * bins + b], d[(j * N + i) * bins + b], "symmetric");

    // Spot-check a value against the reference (symmetrize then linear).
    int i = 0, j = 2, b = 1;
    float ref = lb[b];
    for (int c = 0; c < tz; ++c) ref += lw[b * tz + c] * (z[(i * N + j) * tz + c] + z[(j * N + i) * tz + c]);
    expect_eq_f(d[(i * N + j) * bins + b], ref, "distogram value");
}

int main() {
    std::printf("== test_trunk ==\n");
    return boltztest::run_all();
}
