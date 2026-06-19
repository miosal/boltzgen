// Equivalence test for attention with pair bias vs an independent reference.
#include "boltz/attention_pair_bias.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

void fill(std::vector<float>& v, int n, float seed) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * std::sin(seed + 0.4f * i);
}

AttnPairBiasWeights make(int c_s, int c_z, int H) {
    AttnPairBiasWeights w;
    w.c_s = c_s; w.c_z = c_z; w.num_heads = H; w.inf = 1e6f;
    fill(w.q_w, c_s * c_s, 1); fill(w.q_b, c_s, 1.5f);
    fill(w.k_w, c_s * c_s, 2); fill(w.v_w, c_s * c_s, 3); fill(w.g_w, c_s * c_s, 4);
    w.z_norm_w.assign(c_z, 1.0f); w.z_norm_b.assign(c_z, 0.0f);
    fill(w.z_lin_w, H * c_z, 5); fill(w.o_w, c_s * c_s, 6);
    return w;
}

// Independent reference.
std::vector<float> reference(const std::vector<float>& s, const std::vector<float>& z,
                             const std::vector<float>& mask, int N,
                             const AttnPairBiasWeights& w) {
    const int c_s = w.c_s, c_z = w.c_z, H = w.num_heads, hd = c_s / H;
    auto lin = [&](const float* x, int in, int out, const std::vector<float>& W,
                   const std::vector<float>* b) {
        std::vector<float> y(out);
        for (int o = 0; o < out; ++o) { float a = b ? (*b)[o] : 0; for (int i = 0; i < in; ++i) a += W[o * in + i] * x[i]; y[o] = a; }
        return y;
    };
    auto ln = [&](const float* x, int D, const std::vector<float>& gw, const std::vector<float>& gb) {
        float mu = 0; for (int d = 0; d < D; ++d) mu += x[d]; mu /= D;
        float var = 0; for (int d = 0; d < D; ++d) var += (x[d] - mu) * (x[d] - mu); var /= D;
        float iv = 1.0f / std::sqrt(var + 1e-5f);
        std::vector<float> y(D); for (int d = 0; d < D; ++d) y[d] = (x[d] - mu) * iv * gw[d] + gb[d];
        return y;
    };
    auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    const float scale = 1.0f / std::sqrt((float)hd);

    std::vector<std::vector<float>> q(N), k(N), v(N), g(N);
    for (int i = 0; i < N; ++i) {
        q[i] = lin(&s[i * c_s], c_s, c_s, w.q_w, &w.q_b);
        k[i] = lin(&s[i * c_s], c_s, c_s, w.k_w, nullptr);
        v[i] = lin(&s[i * c_s], c_s, c_s, w.v_w, nullptr);
        g[i] = lin(&s[i * c_s], c_s, c_s, w.g_w, nullptr);
        for (float& e : g[i]) e = sig(e);
    }
    std::vector<float> out(N * c_s, 0.0f);
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < N; ++i) {
            std::vector<float> sc(N);
            float mx = -1e30f;
            for (int j = 0; j < N; ++j) {
                float dot = 0; for (int d = 0; d < hd; ++d) dot += q[i][h * hd + d] * k[j][h * hd + d];
                auto zn = ln(&z[(i * N + j) * c_z], c_z, w.z_norm_w, w.z_norm_b);
                auto bl = lin(zn.data(), c_z, H, w.z_lin_w, nullptr);
                sc[j] = dot * scale + bl[h] + (1.0f - mask[i * N + j]) * (-w.inf);
                mx = std::max(mx, sc[j]);
            }
            float sum = 0; for (int j = 0; j < N; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
            for (int j = 0; j < N; ++j) { float p = sc[j] / sum; for (int d = 0; d < hd; ++d) out[i * c_s + h * hd + d] += p * v[j][h * hd + d]; }
        }
    for (int i = 0; i < N * c_s; ++i) out[i] *= g[i / c_s][i % c_s];
    std::vector<float> res(N * c_s);
    for (int i = 0; i < N; ++i) { auto r = lin(&out[i * c_s], c_s, c_s, w.o_w, nullptr); for (int d = 0; d < c_s; ++d) res[i * c_s + d] = r[d]; }
    return res;
}

}  // namespace

BOLTZ_TEST(attention_pair_bias_matches_reference) {
    const int N = 3, c_s = 4, c_z = 3, H = 2;
    auto w = make(c_s, c_z, H);
    std::vector<float> s(N * c_s), z(N * N * c_z), mask(N * N, 1.0f);
    for (int i = 0; i < N * c_s; ++i) s[i] = 0.2f * std::sin(0.3f + i);
    for (int i = 0; i < N * N * c_z; ++i) z[i] = 0.15f * std::cos(0.2f + i);
    mask[0 * N + 2] = 0.0f;  // mask one key

    auto got = attention_pair_bias(s, z, mask, N, w);
    auto ref = reference(s, z, mask, N, w);
    expect_eq_i(static_cast<long>(got.size()), N * c_s, "shape");
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-5f)
            throw AssertFailure("mismatch at " + std::to_string(i));
}

BOLTZ_TEST(cross_equals_self_when_same_source) {
    const int N = 3, c_s = 4, c_z = 3, H = 2;
    auto w = make(c_s, c_z, H);
    std::vector<float> s(N * c_s), z(N * N * c_z), mask(N * N, 1.0f);
    for (int i = 0; i < N * c_s; ++i) s[i] = 0.2f * std::sin(0.3f + i);
    for (int i = 0; i < N * N * c_z; ++i) z[i] = 0.15f * std::cos(0.2f + i);

    auto self = attention_pair_bias(s, z, mask, N, w);
    auto cross = attention_pair_bias_cross(s, s, z, mask, N, N, w);
    for (size_t i = 0; i < self.size(); ++i) expect_eq_f(self[i], cross[i], "self==cross");
}

BOLTZ_TEST(cross_attention_different_lengths) {
    // Nq=2 queries attend to Nk=4 keys; per-head bias supplied directly.
    const int Nq = 2, Nk = 4, c_s = 4, H = 2;
    AttnPairBiasWeights w;
    w.c_s = c_s; w.c_z = 0; w.num_heads = H; w.compute_pair_bias = false; w.inf = 1e6f;
    fill(w.q_w, c_s * c_s, 1); fill(w.q_b, c_s, 1.5f); fill(w.k_w, c_s * c_s, 2);
    fill(w.v_w, c_s * c_s, 3); fill(w.g_w, c_s * c_s, 4); fill(w.o_w, c_s * c_s, 6);

    std::vector<float> qsrc(Nq * c_s), kvsrc(Nk * c_s), bias(Nq * Nk * H), mask(Nq * Nk, 1.0f);
    for (int i = 0; i < Nq * c_s; ++i) qsrc[i] = 0.2f * std::sin(i);
    for (int i = 0; i < Nk * c_s; ++i) kvsrc[i] = 0.15f * std::cos(i);
    for (int i = 0; i < Nq * Nk * H; ++i) bias[i] = 0.05f * std::sin(0.3f + i);

    auto out = attention_pair_bias_cross(qsrc, kvsrc, bias, mask, Nq, Nk, w);
    expect_eq_i(static_cast<long>(out.size()), Nq * c_s, "Nq*c_s output");
    for (float val : out) expect_true(std::isfinite(val), "finite");

    // Reference: independent attention.
    const int hd = c_s / H;
    const float scale = 1.0f / std::sqrt((float)hd);
    auto lin = [&](const float* x, const std::vector<float>& W, const std::vector<float>* b) {
        std::vector<float> y(c_s); for (int o = 0; o < c_s; ++o) { float a = b ? (*b)[o] : 0; for (int i = 0; i < c_s; ++i) a += W[o * c_s + i] * x[i]; y[o] = a; } return y;
    };
    auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    std::vector<float> ref(Nq * c_s, 0.0f);
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < Nq; ++i) {
            auto qi = lin(&qsrc[i * c_s], w.q_w, &w.q_b);
            std::vector<float> sc(Nk); float mx = -1e30f;
            for (int j = 0; j < Nk; ++j) { auto kj = lin(&kvsrc[j * c_s], w.k_w, nullptr); float d = 0; for (int t = 0; t < hd; ++t) d += qi[h * hd + t] * kj[h * hd + t]; sc[j] = d * scale + bias[(i * Nk + j) * H + h]; mx = std::max(mx, sc[j]); }
            float sum = 0; for (int j = 0; j < Nk; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
            for (int j = 0; j < Nk; ++j) { auto vj = lin(&kvsrc[j * c_s], w.v_w, nullptr); float p = sc[j] / sum; for (int t = 0; t < hd; ++t) ref[i * c_s + h * hd + t] += p * vj[h * hd + t]; }
        }
    for (int i = 0; i < Nq; ++i) { auto gi = lin(&qsrc[i * c_s], w.g_w, nullptr); for (int d = 0; d < c_s; ++d) ref[i * c_s + d] *= sig(gi[d]); }
    std::vector<float> res(Nq * c_s);
    for (int i = 0; i < Nq; ++i) { auto r = lin(&ref[i * c_s], w.o_w, nullptr); for (int d = 0; d < c_s; ++d) res[i * c_s + d] = r[d]; }
    for (int i = 0; i < Nq * c_s; ++i) if (std::fabs(out[i] - res[i]) > 1e-5f) throw AssertFailure("cross mismatch");
}

int main() {
    std::printf("== test_attention_pair_bias ==\n");
    return boltztest::run_all();
}
