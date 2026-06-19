#include "boltz/attention_pair_bias.hpp"

#include <cmath>

namespace boltz {
namespace {

void linear(const float* x, int in, int out, const std::vector<float>& W,
            const std::vector<float>* b, float* y) {
    for (int o = 0; o < out; ++o) {
        float s = b ? (*b)[o] : 0.0f;
        for (int i = 0; i < in; ++i) s += W[o * in + i] * x[i];
        y[o] = s;
    }
}

void layer_norm(const float* x, int D, const std::vector<float>& w,
                const std::vector<float>& b, float* out) {
    float mean = 0;
    for (int d = 0; d < D; ++d) mean += x[d];
    mean /= D;
    float var = 0;
    for (int d = 0; d < D; ++d) { const float v = x[d] - mean; var += v * v; }
    var /= D;
    const float inv = 1.0f / std::sqrt(var + 1e-5f);
    for (int d = 0; d < D; ++d) out[d] = (x[d] - mean) * inv * w[d] + b[d];
}

float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

}  // namespace

std::vector<float> attention_pair_bias(const std::vector<float>& s,
                                       const std::vector<float>& z,
                                       const std::vector<float>& mask, int N,
                                       const AttnPairBiasWeights& w) {
    const int c_s = w.c_s, c_z = w.c_z, H = w.num_heads;
    const int hd = c_s / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // Projections q (with bias), k, v, g (no bias).
    std::vector<float> q((size_t)N * c_s), k((size_t)N * c_s), v((size_t)N * c_s),
        g((size_t)N * c_s);
    for (int i = 0; i < N; ++i) {
        linear(&s[i * c_s], c_s, c_s, w.q_w, &w.q_b, &q[i * c_s]);
        linear(&s[i * c_s], c_s, c_s, w.k_w, nullptr, &k[i * c_s]);
        linear(&s[i * c_s], c_s, c_s, w.v_w, nullptr, &v[i * c_s]);
        linear(&s[i * c_s], c_s, c_s, w.g_w, nullptr, &g[i * c_s]);
        for (int d = 0; d < c_s; ++d) g[i * c_s + d] = sigmoidf(g[i * c_s + d]);
    }

    // Pair bias: LayerNorm(c_z) -> Linear(c_z -> H). bias[i][j][h].
    std::vector<float> bias((size_t)N * N * H);
    std::vector<float> zn(c_z);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            layer_norm(&z[(i * N + j) * c_z], c_z, w.z_norm_w, w.z_norm_b, zn.data());
            linear(zn.data(), c_z, H, w.z_lin_w, nullptr, &bias[(i * N + j) * H]);
        }

    // Per-head attention.
    std::vector<float> o((size_t)N * c_s, 0.0f);
    std::vector<float> scores(N);
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < N; ++i) {
            float mx = -1e30f;
            for (int j = 0; j < N; ++j) {
                float dot = 0;
                for (int d = 0; d < hd; ++d)
                    dot += q[i * c_s + h * hd + d] * k[j * c_s + h * hd + d];
                float sc = dot * scale + bias[(i * N + j) * H + h] +
                           (1.0f - mask[i * N + j]) * (-w.inf);
                scores[j] = sc;
                if (sc > mx) mx = sc;
            }
            float sum = 0;
            for (int j = 0; j < N; ++j) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
            for (int j = 0; j < N; ++j) {
                const float p = scores[j] / sum;
                for (int d = 0; d < hd; ++d)
                    o[i * c_s + h * hd + d] += p * v[j * c_s + h * hd + d];
            }
        }
    }

    // Gate, then output projection.
    for (size_t i = 0; i < o.size(); ++i) o[i] *= g[i];
    std::vector<float> out((size_t)N * c_s);
    for (int i = 0; i < N; ++i)
        linear(&o[i * c_s], c_s, c_s, w.o_w, nullptr, &out[i * c_s]);
    return out;
}

}  // namespace boltz
