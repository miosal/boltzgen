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

std::vector<float> attention_pair_bias_cross(const std::vector<float>& q_src,
                                             const std::vector<float>& kv_src,
                                             const std::vector<float>& z,
                                             const std::vector<float>& mask, int Nq, int Nk,
                                             const AttnPairBiasWeights& w) {
    const int c_s = w.c_s, c_z = w.c_z, H = w.num_heads;
    const int hd = c_s / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // q, g from queries; k, v from keys/values.
    std::vector<float> q((size_t)Nq * c_s), g((size_t)Nq * c_s);
    for (int i = 0; i < Nq; ++i) {
        linear(&q_src[i * c_s], c_s, c_s, w.q_w, &w.q_b, &q[i * c_s]);
        linear(&q_src[i * c_s], c_s, c_s, w.g_w, nullptr, &g[i * c_s]);
        for (int d = 0; d < c_s; ++d) g[i * c_s + d] = sigmoidf(g[i * c_s + d]);
    }
    std::vector<float> k((size_t)Nk * c_s), v((size_t)Nk * c_s);
    for (int j = 0; j < Nk; ++j) {
        linear(&kv_src[j * c_s], c_s, c_s, w.k_w, nullptr, &k[j * c_s]);
        linear(&kv_src[j * c_s], c_s, c_s, w.v_w, nullptr, &v[j * c_s]);
    }

    // Pair bias bias[i][j][h]: computed from z, or z is already per-head bias.
    std::vector<float> bias((size_t)Nq * Nk * H);
    if (w.compute_pair_bias) {
        std::vector<float> zn(c_z);
        for (int i = 0; i < Nq; ++i)
            for (int j = 0; j < Nk; ++j) {
                layer_norm(&z[(i * Nk + j) * c_z], c_z, w.z_norm_w, w.z_norm_b, zn.data());
                linear(zn.data(), c_z, H, w.z_lin_w, nullptr, &bias[(i * Nk + j) * H]);
            }
    } else {
        for (size_t i = 0; i < bias.size(); ++i) bias[i] = z[i];
    }

    std::vector<float> o((size_t)Nq * c_s, 0.0f);
    std::vector<float> scores(Nk);
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < Nq; ++i) {
            float mx = -1e30f;
            for (int j = 0; j < Nk; ++j) {
                float dot = 0;
                for (int d = 0; d < hd; ++d)
                    dot += q[i * c_s + h * hd + d] * k[j * c_s + h * hd + d];
                float sc = dot * scale + bias[(i * Nk + j) * H + h] +
                           (1.0f - mask[i * Nk + j]) * (-w.inf);
                scores[j] = sc;
                if (sc > mx) mx = sc;
            }
            float sum = 0;
            for (int j = 0; j < Nk; ++j) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
            for (int j = 0; j < Nk; ++j) {
                const float p = scores[j] / sum;
                for (int d = 0; d < hd; ++d)
                    o[i * c_s + h * hd + d] += p * v[j * c_s + h * hd + d];
            }
        }
    }

    for (size_t i = 0; i < o.size(); ++i) o[i] *= g[i];
    std::vector<float> out((size_t)Nq * c_s);
    for (int i = 0; i < Nq; ++i)
        linear(&o[i * c_s], c_s, c_s, w.o_w, nullptr, &out[i * c_s]);
    return out;
}

std::vector<float> attention_pair_bias(const std::vector<float>& s,
                                       const std::vector<float>& z,
                                       const std::vector<float>& mask, int N,
                                       const AttnPairBiasWeights& w) {
    // Self-attention is cross-attention with the same query and key source.
    return attention_pair_bias_cross(s, s, z, mask, N, N, w);
}

}  // namespace boltz
