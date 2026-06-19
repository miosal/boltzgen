// Test for the affinity head: cross-pair pooling + ReLU MLPs vs an independent
// reference.
#include "boltz/affinity.hpp"
#include "test_framework.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

LinearB mk(int in, int out, float seed) {
    LinearB L; L.in = in; L.out = out; L.w.resize(in * out); L.b.resize(out);
    for (int i = 0; i < in * out; ++i) L.w[i] = 0.1f * std::sin(seed + 0.3f * i);
    for (int o = 0; o < out; ++o) L.b[o] = 0.05f * std::cos(seed + o);
    return L;
}
std::vector<float> ap(const std::vector<float>& x, const LinearB& L, bool relu) {
    std::vector<float> y(L.out);
    for (int o = 0; o < L.out; ++o) { float s = L.b[o]; for (int i = 0; i < L.in; ++i) s += L.w[o * L.in + i] * x[i]; y[o] = relu ? std::max(0.0f, s) : s; }
    return y;
}

}  // namespace

BOLTZ_TEST(affinity_head_matches_reference) {
    const int N = 3, tz = 3, ts = 2;
    AffinityWeights w;
    w.token_z = tz; w.token_s = ts;
    w.out_l1 = mk(tz, tz, 1); w.out_l2 = mk(tz, ts, 2);
    w.val_l1 = mk(ts, ts, 3); w.val_l2 = mk(ts, ts, 4); w.val_l3 = mk(ts, 1, 5);
    w.sco_l1 = mk(ts, ts, 6); w.sco_l2 = mk(ts, ts, 7); w.sco_l3 = mk(ts, 1, 8);
    w.binary = mk(1, 1, 9);

    std::vector<float> z(N * N * tz);
    for (int i = 0; i < N * N * tz; ++i) z[i] = 0.2f * std::sin(0.5f + i);
    std::vector<float> lig = {0, 0, 1}, rec = {1, 1, 0};  // token2 is ligand, 0/1 receptor

    auto out = affinity_head(z, lig, rec, N, w);

    // Reference pooling.
    std::vector<float> g(tz, 0.0f);
    float denom = 0;
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        if (i == j) continue;
        float cpm = lig[i] * rec[j] + rec[i] * lig[j] + lig[i] * lig[j];
        if (cpm == 0) continue;
        for (int c = 0; c < tz; ++c) g[c] += z[(i * N + j) * tz + c] * cpm;
        denom += cpm;
    }
    for (int c = 0; c < tz; ++c) g[c] /= (denom + 1e-7f);
    auto h = ap(ap(g, w.out_l1, true), w.out_l2, true);
    auto val = ap(ap(ap(h, w.val_l1, true), w.val_l2, true), w.val_l3, false);
    auto sco = ap(ap(ap(h, w.sco_l1, true), w.sco_l2, true), w.sco_l3, false);
    auto bin = ap(sco, w.binary, false);

    expect_eq_f(out.pred_value, val[0], "value");
    expect_eq_f(out.pred_score, sco[0], "score");
    expect_eq_f(out.logits_binary, bin[0], "binary");
}

int main() {
    std::printf("== test_affinity ==\n");
    return boltztest::run_all();
}
