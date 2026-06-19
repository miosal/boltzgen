// Tests for the confidence heads + aggregated-metric (pLDDT) computation.
#include "boltz/confidence.hpp"
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
}  // namespace

BOLTZ_TEST(confidence_heads_shapes_and_values) {
    const int N = 2, ts = 3, tz = 3, pae = 4, pde = 4, plddt = 5;
    ConfidenceWeights w;
    w.token_s = ts; w.token_z = tz; w.num_pae_bins = pae; w.num_pde_bins = pde; w.num_plddt_bins = plddt;
    fill(w.to_pae, pae * tz, 1); fill(w.to_pde, pde * tz, 2);
    fill(w.to_plddt, plddt * ts, 3); fill(w.to_resolved, 2 * ts, 4);

    std::vector<float> s(N * ts), z(N * N * tz);
    for (int i = 0; i < N * ts; ++i) s[i] = 0.2f * std::sin(i);
    for (int i = 0; i < N * N * tz; ++i) z[i] = 0.15f * std::cos(i);

    auto out = confidence_heads(s, z, N, w);
    expect_eq_i(static_cast<long>(out.pae_logits.size()), N * N * pae, "pae shape");
    expect_eq_i(static_cast<long>(out.pde_logits.size()), N * N * pde, "pde shape");
    expect_eq_i(static_cast<long>(out.plddt_logits.size()), N * plddt, "plddt shape");
    expect_eq_i(static_cast<long>(out.resolved_logits.size()), N * 2, "resolved shape");

    // Spot-check plddt head: Linear(s_0).
    for (int b = 0; b < plddt; ++b) {
        float ref = 0;
        for (int c = 0; c < ts; ++c) ref += w.to_plddt[b * ts + c] * s[c];
        expect_eq_f(out.plddt_logits[b], ref, "plddt linear");
    }
    // PDE uses symmetrized z: head(0,1) == head(1,0).
    for (int b = 0; b < pde; ++b)
        expect_eq_f(out.pde_logits[(0 * N + 1) * pde + b], out.pde_logits[(1 * N + 0) * pde + b], "pde symmetric");
}

BOLTZ_TEST(aggregated_metric_formula) {
    // 1 row, 4 bins, end=1.0 -> centers 0.125,0.375,0.625,0.875.
    const int bins = 4;
    std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
    auto v = aggregated_metric(logits, 1, bins, 1.0f);

    float bw = 1.0f / bins;
    std::vector<float> centers(bins);
    for (int b = 0; b < bins; ++b) centers[b] = 0.5f * bw + b * bw;
    float mx = 3.0f, sum = 0;
    std::vector<float> p(bins);
    for (int b = 0; b < bins; ++b) { p[b] = std::exp(logits[b] - mx); sum += p[b]; }
    float ref = 0;
    for (int b = 0; b < bins; ++b) ref += (p[b] / sum) * centers[b];
    expect_eq_f(v[0], ref, "aggregated metric");
    // Sanity: value lies within (0,1).
    expect_true(v[0] > 0.0f && v[0] < 1.0f, "in range");
}

int main() {
    std::printf("== test_confidence ==\n");
    return boltztest::run_all();
}
