// Test the windowed atom attention: shape, finiteness, and that each window's
// output equals a direct per-window cross-attention call (wiring of the
// windowing + gather + mask).
#include "boltz/atom_transformer.hpp"
#include "boltz/atom_windowing.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {
void fill(std::vector<float>& v, int n, float seed) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * std::sin(seed + 0.3f * i);
}
AttnPairBiasWeights mk(int d, int H) {
    AttnPairBiasWeights w; w.c_s = d; w.c_z = 0; w.num_heads = H; w.compute_pair_bias = false; w.inf = 1e6f;
    fill(w.q_w, d * d, 1); fill(w.q_b, d, 1.5f); fill(w.k_w, d * d, 2);
    fill(w.v_w, d * d, 3); fill(w.g_w, d * d, 4); fill(w.o_w, d * d, 6);
    return w;
}
}  // namespace

BOLTZ_TEST(windowed_atom_attention_matches_per_window) {
    const int N = 8, d = 4, W = 4, H = 8, nH = 2;  // K = 2
    auto attn = mk(d, nH);
    std::vector<float> q(N * d), kv(N * d);
    for (int i = 0; i < N * d; ++i) { q[i] = 0.2f * std::sin(0.3f + i); kv[i] = 0.15f * std::cos(0.2f + i); }

    auto out = windowed_atom_attention(q, kv, N, d, W, H, attn);
    expect_eq_i(static_cast<long>(out.size()), N * d, "shape");
    for (float v : out) expect_true(std::isfinite(v), "finite");

    // Reference: gather keys + mask, run cross-attention per window.
    const int K = N / W;
    auto keys = single_to_keys(kv, N, d, W, H);
    std::vector<float> ones(N, 1.0f);
    auto kvalid = single_to_keys(ones, N, 1, W, H);
    for (int k = 0; k < K; ++k) {
        std::vector<float> qwin(q.begin() + k * W * d, q.begin() + (k + 1) * W * d);
        std::vector<float> kwin(keys.begin() + k * H * d, keys.begin() + (k + 1) * H * d);
        std::vector<float> bias(W * H * nH, 0.0f), mask(W * H);
        for (int wq = 0; wq < W; ++wq) for (int hk = 0; hk < H; ++hk) mask[wq * H + hk] = kvalid[k * H + hk];
        auto ref = attention_pair_bias_cross(qwin, kwin, bias, mask, W, H, attn);
        for (int i = 0; i < W * d; ++i)
            expect_eq_f(out[k * W * d + i], ref[i], "window output");
    }
}

int main() {
    std::printf("== test_atom_transformer ==\n");
    return boltztest::run_all();
}
