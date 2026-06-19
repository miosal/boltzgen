// Wiring equivalence test for triangle attention vs an independent reference.
// (Validates the implementation against the documented interpretation; numeric
// parity with the reference model still needs golden tensors.)
#include "boltz/triangle_attention.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

void fill(std::vector<float>& v, int n, float seed) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * std::sin(seed + 0.37f * i);
}

TriAttnWeights make(int C, int H, int hd) {
    TriAttnWeights w;
    w.c_in = C; w.num_heads = H; w.head_dim = hd; w.inf = 1e9f;
    w.norm_w.assign(C, 1.0f); w.norm_b.assign(C, 0.0f);
    fill(w.tri_w, H * C, 1);
    fill(w.q_w, H * hd * C, 2);
    fill(w.k_w, H * hd * C, 3);
    fill(w.v_w, H * hd * C, 4);
    fill(w.g_w, H * hd * C, 5);
    fill(w.o_w, C * H * hd, 6);
    return w;
}

std::vector<float> reference(const std::vector<float>& xin, const std::vector<float>& min,
                             int N, const TriAttnWeights& w, bool starting) {
    const int C = w.c_in, H = w.num_heads, hd = w.head_dim, HD = H * hd;
    const float scale = 1.0f / std::sqrt((float)hd);
    auto lin = [&](const float* x, int in, int out, const std::vector<float>& W) {
        std::vector<float> y(out, 0.0f);
        for (int o = 0; o < out; ++o) for (int i = 0; i < in; ++i) y[o] += W[o * in + i] * x[i];
        return y;
    };
    auto ln = [&](const float* x) {
        float mu = 0; for (int d = 0; d < C; ++d) mu += x[d]; mu /= C;
        float var = 0; for (int d = 0; d < C; ++d) var += (x[d] - mu) * (x[d] - mu); var /= C;
        float iv = 1.0f / std::sqrt(var + 1e-5f);
        std::vector<float> y(C); for (int d = 0; d < C; ++d) y[d] = (x[d] - mu) * iv;
        return y;
    };
    auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };

    // transpose if ending
    std::vector<float> x(N * N * C), mask(N * N);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        int si = starting ? i : j, sj = starting ? j : i;
        for (int c = 0; c < C; ++c) x[(i * N + j) * C + c] = xin[(si * N + sj) * C + c];
        mask[i * N + j] = min[si * N + sj];
    }
    std::vector<std::vector<float>> xn(N * N);
    for (int p = 0; p < N * N; ++p) xn[p] = ln(&x[p * C]);
    auto out = std::vector<float>(N * N * C, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int qp = 0; qp < N; ++qp) {
            auto qv = lin(xn[i * N + qp].data(), C, HD, w.q_w);
            std::vector<float> ov(HD, 0.0f);
            for (int h = 0; h < H; ++h) {
                std::vector<float> sc(N);
                float mx = -1e30f;
                for (int kp = 0; kp < N; ++kp) {
                    auto kv = lin(xn[i * N + kp].data(), C, HD, w.k_w);
                    auto tb = lin(xn[qp * N + kp].data(), C, H, w.tri_w);
                    float dot = 0; for (int d = 0; d < hd; ++d) dot += qv[h * hd + d] * kv[h * hd + d];
                    sc[kp] = dot * scale + tb[h] + w.inf * (mask[i * N + kp] - 1.0f);
                    mx = std::max(mx, sc[kp]);
                }
                float sum = 0; for (int kp = 0; kp < N; ++kp) { sc[kp] = std::exp(sc[kp] - mx); sum += sc[kp]; }
                for (int kp = 0; kp < N; ++kp) {
                    auto vv = lin(xn[i * N + kp].data(), C, HD, w.v_w);
                    float p = sc[kp] / sum;
                    for (int d = 0; d < hd; ++d) ov[h * hd + d] += p * vv[h * hd + d];
                }
            }
            auto gv = lin(xn[i * N + qp].data(), C, HD, w.g_w);
            for (int d = 0; d < HD; ++d) ov[d] *= sig(gv[d]);
            auto o = lin(ov.data(), HD, C, w.o_w);
            for (int c = 0; c < C; ++c) out[(i * N + qp) * C + c] = o[c];
        }
    if (!starting) {
        std::vector<float> t(N * N * C);
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) for (int c = 0; c < C; ++c)
            t[(j * N + i) * C + c] = out[(i * N + j) * C + c];
        return t;
    }
    return out;
}

std::vector<float> make_x(int N, int C) {
    std::vector<float> x(N * N * C);
    for (int i = 0; i < N * N * C; ++i) x[i] = 0.25f * std::sin(0.6f + 0.3f * i);
    return x;
}

}  // namespace

BOLTZ_TEST(triangle_attention_starting_matches_reference) {
    const int N = 3, C = 4, H = 2, hd = 2;
    auto w = make(C, H, hd);
    auto x = make_x(N, C);
    std::vector<float> mask(N * N, 1.0f);
    mask[0 * N + 2] = 0.0f;
    auto got = triangle_attention(x, mask, N, w, true);
    auto ref = reference(x, mask, N, w, true);
    expect_eq_i(static_cast<long>(got.size()), N * N * C, "shape");
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-5f) throw AssertFailure("start mismatch " + std::to_string(i));
}

BOLTZ_TEST(triangle_attention_ending_matches_reference) {
    const int N = 3, C = 4, H = 2, hd = 2;
    auto w = make(C, H, hd);
    auto x = make_x(N, C);
    std::vector<float> mask(N * N, 1.0f);
    auto got = triangle_attention(x, mask, N, w, false);
    auto ref = reference(x, mask, N, w, false);
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-5f) throw AssertFailure("end mismatch " + std::to_string(i));
}

BOLTZ_TEST(triangle_attention_start_end_differ) {
    const int N = 3, C = 4, H = 2, hd = 2;
    auto w = make(C, H, hd);
    auto x = make_x(N, C);
    std::vector<float> mask(N * N, 1.0f);
    auto s = triangle_attention(x, mask, N, w, true);
    auto e = triangle_attention(x, mask, N, w, false);
    bool differ = false;
    for (size_t i = 0; i < s.size(); ++i) if (std::fabs(s[i] - e[i]) > 1e-6f) { differ = true; break; }
    expect_true(differ, "starting and ending differ");
}

int main() {
    std::printf("== test_triangle_attention ==\n");
    return boltztest::run_all();
}
