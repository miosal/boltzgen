// Equivalence test for triangle multiplication. The implementation is checked
// against an independent from-scratch reference, with special attention to the
// einsum contraction direction (outgoing vs incoming).
#include "boltz/triangle_mult.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

float sig(float v) { return 1.0f / (1.0f + std::exp(-v)); }

TriMulWeights make_weights(int D) {
    TriMulWeights w;
    w.dim = D;
    w.norm_in_w.assign(D, 1.0f);
    w.norm_in_b.assign(D, 0.0f);
    w.norm_out_w.assign(D, 1.0f);
    w.norm_out_b.assign(D, 0.0f);
    auto fill = [](std::vector<float>& v, int n, float seed) {
        v.resize(n);
        for (int i = 0; i < n; ++i) v[i] = 0.1f * std::sin(seed + 0.5f * i);
    };
    fill(w.p_in, 2 * D * D, 1.0f);
    fill(w.g_in, 2 * D * D, 2.0f);
    fill(w.p_out, D * D, 3.0f);
    fill(w.g_out, D * D, 4.0f);
    return w;
}

// Independent reference (fully separate code path).
std::vector<float> reference(const std::vector<float>& x, const std::vector<float>& mask,
                             int N, int D, const TriMulWeights& w, bool incoming) {
    auto ln = [&](std::vector<float> v, const std::vector<float>& gw,
                  const std::vector<float>& gb) {
        float mu = 0; for (float e : v) mu += e; mu /= D;
        float var = 0; for (float e : v) var += (e - mu) * (e - mu); var /= D;
        float inv = 1.0f / std::sqrt(var + 1e-5f);
        for (int d = 0; d < D; ++d) v[d] = (v[d] - mu) * inv * gw[d] + gb[d];
        return v;
    };
    auto lin = [&](const std::vector<float>& v, const std::vector<float>& W, int out) {
        std::vector<float> y(out, 0.0f);
        for (int o = 0; o < out; ++o)
            for (int i = 0; i < D; ++i) y[o] += W[o * D + i] * v[i];
        return y;
    };
    auto row = [&](const std::vector<float>& t, int i, int j) {
        return std::vector<float>(t.begin() + (i * N + j) * D, t.begin() + (i * N + j + 1) * D);
    };

    std::vector<float> xn((size_t)N * N * D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            auto r = ln(row(x, i, j), w.norm_in_w, w.norm_in_b);
            for (int d = 0; d < D; ++d) xn[(i * N + j) * D + d] = r[d];
        }
    std::vector<float> a((size_t)N * N * D), b((size_t)N * N * D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            auto in = row(xn, i, j);
            auto p = lin(in, w.p_in, 2 * D), g = lin(in, w.g_in, 2 * D);
            float m = mask[i * N + j];
            for (int d = 0; d < D; ++d) {
                a[(i * N + j) * D + d] = p[d] * sig(g[d]) * m;
                b[(i * N + j) * D + d] = p[D + d] * sig(g[D + d]) * m;
            }
        }
    std::vector<float> t((size_t)N * N * D, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                for (int d = 0; d < D; ++d) {
                    float av = incoming ? a[(k * N + i) * D + d] : a[(i * N + k) * D + d];
                    float bv = incoming ? b[(k * N + j) * D + d] : b[(j * N + k) * D + d];
                    t[(i * N + j) * D + d] += av * bv;
                }
    std::vector<float> out((size_t)N * N * D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            auto tn = ln(row(t, i, j), w.norm_out_w, w.norm_out_b);
            auto po = lin(tn, w.p_out, D);
            auto go = lin(row(xn, i, j), w.g_out, D);
            for (int d = 0; d < D; ++d) out[(i * N + j) * D + d] = po[d] * sig(go[d]);
        }
    return out;
}

std::vector<float> make_x(int N, int D) {
    std::vector<float> x((size_t)N * N * D);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.3f * std::sin(0.7f + 0.4f * i);
    return x;
}

}  // namespace

BOLTZ_TEST(triangle_mult_outgoing_matches_reference) {
    const int N = 3, D = 4;
    auto w = make_weights(D);
    auto x = make_x(N, D);
    std::vector<float> mask(N * N, 1.0f);
    mask[1 * N + 2] = 0.0f;  // exercise masking
    auto got = triangle_multiplication(x, mask, N, D, w, /*incoming=*/false);
    auto ref = reference(x, mask, N, D, w, false);
    expect_eq_i(static_cast<long>(got.size()), N * N * D, "shape");
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-5f)
            throw AssertFailure("outgoing mismatch at " + std::to_string(i));
}

BOLTZ_TEST(triangle_mult_incoming_matches_reference) {
    const int N = 3, D = 4;
    auto w = make_weights(D);
    auto x = make_x(N, D);
    std::vector<float> mask(N * N, 1.0f);
    auto got = triangle_multiplication(x, mask, N, D, w, /*incoming=*/true);
    auto ref = reference(x, mask, N, D, w, true);
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-5f)
            throw AssertFailure("incoming mismatch at " + std::to_string(i));
}

BOLTZ_TEST(triangle_mult_direction_differs) {
    const int N = 3, D = 4;
    auto w = make_weights(D);
    auto x = make_x(N, D);
    std::vector<float> mask(N * N, 1.0f);
    auto out_o = triangle_multiplication(x, mask, N, D, w, false);
    auto out_i = triangle_multiplication(x, mask, N, D, w, true);
    bool differ = false;
    for (size_t i = 0; i < out_o.size(); ++i)
        if (std::fabs(out_o[i] - out_i[i]) > 1e-6f) { differ = true; break; }
    expect_true(differ, "outgoing and incoming should differ");
}

int main() {
    std::printf("== test_triangle_mult ==\n");
    return boltztest::run_all();
}
