#include "boltz/triangle_mult.hpp"

#include <cmath>

namespace boltz {
namespace {

float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

// LayerNorm over D with affine (population variance, eps 1e-5).
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

// y[out] = W[out,in] @ x[in]  (no bias).
void linear_nobias(const float* x, int in, int out, const std::vector<float>& W,
                   float* y) {
    for (int o = 0; o < out; ++o) {
        float s = 0;
        for (int i = 0; i < in; ++i) s += W[o * in + i] * x[i];
        y[o] = s;
    }
}

}  // namespace

std::vector<float> triangle_multiplication(const std::vector<float>& x,
                                           const std::vector<float>& mask, int N, int D,
                                           const TriMulWeights& w, bool incoming) {
    const int NN = N * N;
    auto at = [&](const std::vector<float>& t, int i, int j) { return &t[(i * N + j) * D]; };

    // norm_in -> xn (also serves as x_in for the output gate).
    std::vector<float> xn((size_t)NN * D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            layer_norm(at(x, i, j), D, w.norm_in_w, w.norm_in_b, &xn[(i * N + j) * D]);

    // a, b = chunk(p_in(xn) * sigmoid(g_in(xn)) * mask, 2)
    std::vector<float> a((size_t)NN * D), b((size_t)NN * D);
    std::vector<float> pin(2 * D), gin(2 * D);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float* in = &xn[(i * N + j) * D];
            linear_nobias(in, D, 2 * D, w.p_in, pin.data());
            linear_nobias(in, D, 2 * D, w.g_in, gin.data());
            const float m = mask[i * N + j];
            for (int d = 0; d < D; ++d) {
                const float v = pin[d] * sigmoidf(gin[d]) * m;
                const float u = pin[D + d] * sigmoidf(gin[D + d]) * m;
                a[(i * N + j) * D + d] = v;
                b[(i * N + j) * D + d] = u;
            }
        }
    }

    // Triangular contraction.
    std::vector<float> t((size_t)NN * D, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float* o = &t[(i * N + j) * D];
            for (int k = 0; k < N; ++k) {
                const float* ad = incoming ? at(a, k, i) : at(a, i, k);
                const float* bd = incoming ? at(b, k, j) : at(b, j, k);
                for (int d = 0; d < D; ++d) o[d] += ad[d] * bd[d];
            }
        }
    }

    // out = p_out(norm_out(t)) * sigmoid(g_out(x_in))
    std::vector<float> out((size_t)NN * D);
    std::vector<float> tn(D), pout(D), gout(D);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            layer_norm(&t[(i * N + j) * D], D, w.norm_out_w, w.norm_out_b, tn.data());
            linear_nobias(tn.data(), D, D, w.p_out, pout.data());
            linear_nobias(&xn[(i * N + j) * D], D, D, w.g_out, gout.data());
            for (int d = 0; d < D; ++d)
                out[(i * N + j) * D + d] = pout[d] * sigmoidf(gout[d]);
        }
    }
    return out;
}

}  // namespace boltz
