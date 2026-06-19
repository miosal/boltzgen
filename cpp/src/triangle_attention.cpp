#include "boltz/triangle_attention.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace boltz {
namespace {

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

void linear_nobias(const float* x, int in, int out, const std::vector<float>& W, float* y) {
    for (int o = 0; o < out; ++o) {
        float s = 0;
        for (int i = 0; i < in; ++i) s += W[o * in + i] * x[i];
        y[o] = s;
    }
}

float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

}  // namespace

std::vector<float> triangle_attention(const std::vector<float>& x_in,
                                      const std::vector<float>& mask_in, int N,
                                      const TriAttnWeights& w, bool starting) {
    const int C = w.c_in, H = w.num_heads, hd = w.head_dim, HD = H * hd;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // Ending node: operate on the transpose, transpose back at the end.
    std::vector<float> x((size_t)N * N * C), mask((size_t)N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const int si = starting ? i : j, sj = starting ? j : i;
            for (int c = 0; c < C; ++c) x[(i * N + j) * C + c] = x_in[(si * N + sj) * C + c];
            mask[i * N + j] = mask_in[si * N + sj];
        }

    // LayerNorm over c_in.
    std::vector<float> xn((size_t)N * N * C);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            layer_norm(&x[(i * N + j) * C], C, w.norm_w, w.norm_b, &xn[(i * N + j) * C]);

    // Triangle bias triBias[q][k][h] = (W_tri xn[q,k])[h], shared across rows.
    std::vector<float> tri((size_t)N * N * H);
    for (int p = 0; p < N; ++p)
        for (int q = 0; q < N; ++q)
            linear_nobias(&xn[(p * N + q) * C], C, H, w.tri_w, &tri[(p * N + q) * H]);

    // q/k/v/g projections for every (row, pos).
    std::vector<float> q((size_t)N * N * HD), k((size_t)N * N * HD), v((size_t)N * N * HD),
        g((size_t)N * N * HD);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const float* in = &xn[(i * N + j) * C];
            linear_nobias(in, C, HD, w.q_w, &q[(i * N + j) * HD]);
            linear_nobias(in, C, HD, w.k_w, &k[(i * N + j) * HD]);
            linear_nobias(in, C, HD, w.v_w, &v[(i * N + j) * HD]);
            linear_nobias(in, C, HD, w.g_w, &g[(i * N + j) * HD]);
        }

    std::vector<float> out((size_t)N * N * C);
    std::vector<float> ovec(HD), scores(N);
    for (int i = 0; i < N; ++i) {            // row
        for (int qpos = 0; qpos < N; ++qpos) {  // query position in row
            std::fill(ovec.begin(), ovec.end(), 0.0f);
            for (int h = 0; h < H; ++h) {
                const float* qh = &q[(i * N + qpos) * HD + h * hd];
                float mx = -1e30f;
                for (int kpos = 0; kpos < N; ++kpos) {
                    const float* kh = &k[(i * N + kpos) * HD + h * hd];
                    float dot = 0;
                    for (int d = 0; d < hd; ++d) dot += qh[d] * kh[d];
                    float sc = dot * scale + tri[(qpos * N + kpos) * H + h] +
                               w.inf * (mask[i * N + kpos] - 1.0f);
                    scores[kpos] = sc;
                    if (sc > mx) mx = sc;
                }
                float sum = 0;
                for (int kpos = 0; kpos < N; ++kpos) { scores[kpos] = std::exp(scores[kpos] - mx); sum += scores[kpos]; }
                for (int kpos = 0; kpos < N; ++kpos) {
                    const float p = scores[kpos] / sum;
                    const float* vh = &v[(i * N + kpos) * HD + h * hd];
                    for (int d = 0; d < hd; ++d) ovec[h * hd + d] += p * vh[d];
                }
            }
            // Gate + output projection.
            const float* gv = &g[(i * N + qpos) * HD];
            for (int d = 0; d < HD; ++d) ovec[d] *= sigmoidf(gv[d]);
            linear_nobias(ovec.data(), HD, C, w.o_w, &out[(i * N + qpos) * C]);
        }
    }

    // Transpose back for ending node.
    if (!starting) {
        std::vector<float> t((size_t)N * N * C);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                for (int c = 0; c < C; ++c) t[(j * N + i) * C + c] = out[(i * N + j) * C + c];
        return t;
    }
    return out;
}

}  // namespace boltz
