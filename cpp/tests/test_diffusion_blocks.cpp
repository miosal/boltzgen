// Equivalence tests for AdaLN and ConditionedTransitionBlock vs independent refs.
#include "boltz/diffusion_blocks.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

void fill(std::vector<float>& v, int n, float seed) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * std::sin(seed + 0.33f * i);
}
float sig(float v) { return 1.0f / (1.0f + std::exp(-v)); }
float silu(float v) { return v / (1.0f + std::exp(-v)); }

std::vector<float> lnf(const std::vector<float>& x, const std::vector<float>* w) {
    int D = x.size();
    float mu = 0; for (float v : x) mu += v; mu /= D;
    float var = 0; for (float v : x) var += (v - mu) * (v - mu); var /= D;
    float iv = 1.0f / std::sqrt(var + 1e-5f);
    std::vector<float> y(D);
    for (int d = 0; d < D; ++d) { y[d] = (x[d] - mu) * iv; if (w) y[d] *= (*w)[d]; }
    return y;
}
std::vector<float> linf(const std::vector<float>& x, const std::vector<float>& W, int out,
                        const std::vector<float>* b) {
    int in = x.size();
    std::vector<float> y(out);
    for (int o = 0; o < out; ++o) { float s = b ? (*b)[o] : 0; for (int i = 0; i < in; ++i) s += W[o * in + i] * x[i]; y[o] = s; }
    return y;
}

AdaLNWeights make_adaln(int dim, int dc) {
    AdaLNWeights w; w.dim = dim; w.dim_cond = dc;
    fill(w.s_norm_w, dc, 1);
    fill(w.s_scale_w, dim * dc, 2); fill(w.s_scale_b, dim, 3);
    fill(w.s_bias_w, dim * dc, 4);
    return w;
}

}  // namespace

BOLTZ_TEST(adaln_matches_reference) {
    const int dim = 4, dc = 4;
    auto w = make_adaln(dim, dc);
    std::vector<float> a(dim), s(dc);
    for (int i = 0; i < dim; ++i) a[i] = 0.3f * std::sin(i + 1);
    for (int i = 0; i < dc; ++i) s[i] = 0.2f * std::cos(i + 1);

    auto got = adaln(a, s, w);
    auto an = lnf(a, nullptr);
    auto sn = lnf(s, &w.s_norm_w);
    auto sc = linf(sn, w.s_scale_w, dim, &w.s_scale_b);
    auto bi = linf(sn, w.s_bias_w, dim, nullptr);
    for (int d = 0; d < dim; ++d) expect_eq_f(got[d], sig(sc[d]) * an[d] + bi[d], "adaln");
}

BOLTZ_TEST(conditioned_transition_matches_reference) {
    const int dim = 4, dc = 4, di = 8;
    ConditionedTransitionWeights w;
    w.dim = dim; w.dim_cond = dc; w.dim_inner = di;
    w.adaln = make_adaln(dim, dc);
    fill(w.swish_gate_w, 2 * di * dim, 5);
    fill(w.a_to_b_w, di * dim, 6);
    fill(w.b_to_a_w, dim * di, 7);
    fill(w.op_w, dim * dc, 8); fill(w.op_b, dim, 9);

    std::vector<float> a(dim), s(dc);
    for (int i = 0; i < dim; ++i) a[i] = 0.25f * std::sin(i + 2);
    for (int i = 0; i < dc; ++i) s[i] = 0.15f * std::cos(i + 3);

    auto got = conditioned_transition_block(a, s, w);

    auto a2 = adaln(a, s, w.adaln);
    auto sg = linf(a2, w.swish_gate_w, 2 * di, nullptr);
    auto atb = linf(a2, w.a_to_b_w, di, nullptr);
    std::vector<float> b(di);
    for (int i = 0; i < di; ++i) b[i] = (silu(sg[di + i]) * sg[i]) * atb[i];
    auto bta = linf(b, w.b_to_a_w, dim, nullptr);
    auto op = linf(s, w.op_w, dim, &w.op_b);
    for (int d = 0; d < dim; ++d) expect_eq_f(got[d], sig(op[d]) * bta[d], "cond transition");
}

BOLTZ_TEST(fourier_embedding_matches_reference) {
    std::vector<float> weight = {0.5f, -1.0f, 2.0f}, bias = {0.1f, 0.2f, -0.3f};
    const float t = 0.7f;
    auto got = fourier_embedding(t, weight, bias);
    const float two_pi = 6.283185307179586f;
    for (int d = 0; d < 3; ++d)
        expect_eq_f(got[d], std::cos(two_pi * (bias[d] + t * weight[d])), "fourier");
}

int main() {
    std::printf("== test_diffusion_blocks ==\n");
    return boltztest::run_all();
}
