// Wiring test for the DiffusionTransformer: layer + stack vs independent
// composition of the (already-tested) adaln / attention / conditioned-transition.
#include "boltz/diffusion_transformer.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

void fill(std::vector<float>& v, int n, float seed) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.05f * std::sin(seed + 0.27f * i);
}
float sig(float v) { return 1.0f / (1.0f + std::exp(-v)); }

AdaLNWeights mk_adaln(int dim, int dc, float s) {
    AdaLNWeights w; w.dim = dim; w.dim_cond = dc;
    fill(w.s_norm_w, dc, s + 1); fill(w.s_scale_w, dim * dc, s + 2);
    fill(w.s_scale_b, dim, s + 3); fill(w.s_bias_w, dim * dc, s + 4);
    return w;
}
ConditionedTransitionWeights mk_ct(int dim, int dc, int di, float s) {
    ConditionedTransitionWeights w; w.dim = dim; w.dim_cond = dc; w.dim_inner = di;
    w.adaln = mk_adaln(dim, dc, s + 10);
    fill(w.swish_gate_w, 2 * di * dim, s + 1); fill(w.a_to_b_w, di * dim, s + 2);
    fill(w.b_to_a_w, dim * di, s + 3); fill(w.op_w, dim * dc, s + 4); fill(w.op_b, dim, s + 5);
    return w;
}
DiffusionTransformerLayerWeights mk_layer(int dim, int dc, int H, float s) {
    DiffusionTransformerLayerWeights w; w.dim = dim; w.dim_cond = dc; w.num_heads = H;
    w.adaln = mk_adaln(dim, dc, s);
    w.attn.c_s = dim; w.attn.c_z = 0; w.attn.num_heads = H; w.attn.compute_pair_bias = false; w.attn.inf = 1e6f;
    fill(w.attn.q_w, dim * dim, s + 20); fill(w.attn.q_b, dim, s + 20.5f);
    fill(w.attn.k_w, dim * dim, s + 21); fill(w.attn.v_w, dim * dim, s + 22);
    fill(w.attn.g_w, dim * dim, s + 23); fill(w.attn.o_w, dim * dim, s + 24);
    fill(w.op_w, dim * dc, s + 30); fill(w.op_b, dim, s + 31);
    w.transition = mk_ct(dim, dc, dim * 2, s + 40);
    return w;
}

}  // namespace

BOLTZ_TEST(diffusion_transformer_layer_matches_composition) {
    const int N = 3, dim = 4, dc = 4, H = 2;
    auto w = mk_layer(dim, dc, H, 0);
    std::vector<float> a(N * dim), s(N * dc), bias(N * N * H), tmask(N, 1.0f);
    for (int i = 0; i < N * dim; ++i) a[i] = 0.2f * std::sin(0.3f + i);
    for (int i = 0; i < N * dc; ++i) s[i] = 0.15f * std::cos(0.2f + i);
    for (int i = 0; i < N * N * H; ++i) bias[i] = 0.1f * std::sin(0.5f + i);
    tmask[2] = 0.0f;

    auto got = diffusion_transformer_layer(a, s, bias, tmask, N, w);

    // Independent composition.
    std::vector<float> b(N * dim);
    for (int i = 0; i < N; ++i) {
        auto bi = adaln(std::vector<float>(a.begin() + i * dim, a.begin() + (i + 1) * dim),
                        std::vector<float>(s.begin() + i * dc, s.begin() + (i + 1) * dc), w.adaln);
        for (int d = 0; d < dim; ++d) b[i * dim + d] = bi[d];
    }
    std::vector<float> mask2d(N * N);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) mask2d[i * N + j] = tmask[j];
    auto attn = attention_pair_bias(b, bias, mask2d, N, w.attn);
    std::vector<float> ref = a;
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < dim; ++d) {
            float acc = w.op_b[d];
            for (int c = 0; c < dc; ++c) acc += w.op_w[d * dc + c] * s[i * dc + c];
            ref[i * dim + d] += sig(acc) * attn[i * dim + d];
        }
    for (int i = 0; i < N; ++i) {
        auto t = conditioned_transition_block(
            std::vector<float>(ref.begin() + i * dim, ref.begin() + (i + 1) * dim),
            std::vector<float>(s.begin() + i * dc, s.begin() + (i + 1) * dc), w.transition);
        for (int d = 0; d < dim; ++d) ref[i * dim + d] += t[d];
    }
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-4f) throw AssertFailure("layer mismatch " + std::to_string(i));
}

BOLTZ_TEST(diffusion_transformer_stack_runs) {
    const int N = 3, dim = 4, dc = 4, H = 2;
    std::vector<DiffusionTransformerLayerWeights> layers = {mk_layer(dim, dc, H, 0), mk_layer(dim, dc, H, 100)};
    std::vector<float> a(N * dim), s(N * dc), tmask(N, 1.0f);
    std::vector<std::vector<float>> biases(2, std::vector<float>(N * N * H));
    for (int i = 0; i < N * dim; ++i) a[i] = 0.2f * std::sin(i);
    for (int i = 0; i < N * dc; ++i) s[i] = 0.1f * std::cos(i);
    for (int l = 0; l < 2; ++l) for (int i = 0; i < N * N * H; ++i) biases[l][i] = 0.05f * std::sin(l + i);

    auto got = diffusion_transformer(a, s, biases, tmask, N, layers);
    auto l0 = diffusion_transformer_layer(a, s, biases[0], tmask, N, layers[0]);
    auto l1 = diffusion_transformer_layer(l0, s, biases[1], tmask, N, layers[1]);
    for (size_t i = 0; i < l1.size(); ++i)
        if (std::fabs(got[i] - l1[i]) > 1e-5f) throw AssertFailure("stack mismatch");
    // finite output of right shape
    expect_eq_i(static_cast<long>(got.size()), N * dim, "shape");
    for (float v : got) expect_true(std::isfinite(v), "finite");
}

int main() {
    std::printf("== test_diffusion_transformer ==\n");
    return boltztest::run_all();
}
