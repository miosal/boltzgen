// Equivalence test for OuterProductMean vs an independent reference.
#include "boltz/outer_product_mean.hpp"
#include "test_framework.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

void fill(std::vector<float>& v, int n, float seed) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * std::sin(seed + 0.4f * i);
}

OuterProductMeanWeights make(int ci, int ch, int co) {
    OuterProductMeanWeights w;
    w.c_in = ci; w.c_hidden = ch; w.c_out = co;
    w.norm_w.assign(ci, 1.0f); w.norm_b.assign(ci, 0.0f);
    fill(w.proj_a, ch * ci, 1);
    fill(w.proj_b, ch * ci, 2);
    fill(w.proj_o_w, co * ch * ch, 3);
    fill(w.proj_o_b, co, 4);
    return w;
}

std::vector<float> reference(const std::vector<float>& m, const std::vector<float>& mask,
                             int S, int N, const OuterProductMeanWeights& w) {
    const int ci = w.c_in, ch = w.c_hidden, co = w.c_out, ch2 = ch * ch;
    auto ln = [&](const float* x) {
        float mu = 0; for (int d = 0; d < ci; ++d) mu += x[d]; mu /= ci;
        float var = 0; for (int d = 0; d < ci; ++d) var += (x[d] - mu) * (x[d] - mu); var /= ci;
        float iv = 1.0f / std::sqrt(var + 1e-5f);
        std::vector<float> y(ci); for (int d = 0; d < ci; ++d) y[d] = (x[d] - mu) * iv;
        return y;
    };
    auto proj = [&](const std::vector<float>& x, const std::vector<float>& W, int out) {
        std::vector<float> y(out, 0.0f);
        for (int o = 0; o < out; ++o) for (int i = 0; i < (int)x.size(); ++i) y[o] += W[o * x.size() + i] * x[i];
        return y;
    };
    std::vector<std::vector<float>> a(S * N), b(S * N);
    for (int s = 0; s < S; ++s) for (int i = 0; i < N; ++i) {
        auto n = ln(&m[(s * N + i) * ci]);
        a[s * N + i] = proj(n, w.proj_a, ch);
        b[s * N + i] = proj(n, w.proj_b, ch);
        float mk = mask[s * N + i];
        for (int c = 0; c < ch; ++c) { a[s * N + i][c] *= mk; b[s * N + i][c] *= mk; }
    }
    std::vector<float> out(N * N * co);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        std::vector<float> flat(ch2, 0.0f);
        float ms = 0;
        for (int s = 0; s < S; ++s) {
            for (int c = 0; c < ch; ++c) for (int d = 0; d < ch; ++d) flat[c * ch + d] += a[s * N + i][c] * b[s * N + j][d];
            ms += mask[s * N + i] * mask[s * N + j];
        }
        float denom = std::max(1.0f, ms);
        for (float& f : flat) f /= denom;
        auto o = proj(flat, w.proj_o_w, co);
        for (int c = 0; c < co; ++c) out[(i * N + j) * co + c] = o[c] + w.proj_o_b[c];
    }
    return out;
}

}  // namespace

BOLTZ_TEST(outer_product_mean_matches_reference) {
    const int S = 3, N = 2, ci = 3, ch = 2, co = 3;
    auto w = make(ci, ch, co);
    std::vector<float> m(S * N * ci), mask(S * N, 1.0f);
    for (int i = 0; i < S * N * ci; ++i) m[i] = 0.2f * std::sin(0.5f + i);
    mask[1 * N + 0] = 0.0f;  // drop one (seq,res) entry

    auto got = outer_product_mean(m, mask, S, N, w);
    auto ref = reference(m, mask, S, N, w);
    expect_eq_i(static_cast<long>(got.size()), N * N * co, "shape");
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-5f)
            throw AssertFailure("mismatch at " + std::to_string(i));
}

int main() {
    std::printf("== test_outer_product_mean ==\n");
    return boltztest::run_all();
}
