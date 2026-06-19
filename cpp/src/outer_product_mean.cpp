#include "boltz/outer_product_mean.hpp"

#include <algorithm>
#include <cmath>

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

void linear(const float* x, int in, int out, const std::vector<float>& W,
            const std::vector<float>* b, float* y) {
    for (int o = 0; o < out; ++o) {
        float s = b ? (*b)[o] : 0.0f;
        for (int i = 0; i < in; ++i) s += W[o * in + i] * x[i];
        y[o] = s;
    }
}

}  // namespace

std::vector<float> outer_product_mean(const std::vector<float>& m,
                                      const std::vector<float>& mask, int S, int N,
                                      const OuterProductMeanWeights& w) {
    const int ci = w.c_in, ch = w.c_hidden, co = w.c_out;

    // a, b: [S, N, c_hidden] = proj(norm(m)) * mask.
    std::vector<float> a((size_t)S * N * ch), b((size_t)S * N * ch);
    std::vector<float> nm(ci);
    for (int s = 0; s < S; ++s) {
        for (int i = 0; i < N; ++i) {
            const float* mi = &m[((size_t)s * N + i) * ci];
            layer_norm(mi, ci, w.norm_w, w.norm_b, nm.data());
            const float mk = mask[s * N + i];
            linear(nm.data(), ci, ch, w.proj_a, nullptr, &a[((size_t)s * N + i) * ch]);
            linear(nm.data(), ci, ch, w.proj_b, nullptr, &b[((size_t)s * N + i) * ch]);
            for (int c = 0; c < ch; ++c) {
                a[((size_t)s * N + i) * ch + c] *= mk;
                b[((size_t)s * N + i) * ch + c] *= mk;
            }
        }
    }

    const int ch2 = ch * ch;
    std::vector<float> out((size_t)N * N * co);
    std::vector<float> flat(ch2);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // Outer-product accumulate over sequences + mask sum.
            std::fill(flat.begin(), flat.end(), 0.0f);
            float mask_sum = 0;
            for (int s = 0; s < S; ++s) {
                const float* as = &a[((size_t)s * N + i) * ch];
                const float* bs = &b[((size_t)s * N + j) * ch];
                for (int c = 0; c < ch; ++c)
                    for (int d = 0; d < ch; ++d) flat[c * ch + d] += as[c] * bs[d];
                mask_sum += mask[s * N + i] * mask[s * N + j];
            }
            const float denom = std::max(1.0f, mask_sum);
            for (int c = 0; c < ch2; ++c) flat[c] /= denom;
            linear(flat.data(), ch2, co, w.proj_o_w, &w.proj_o_b, &out[((size_t)i * N + j) * co]);
        }
    }
    return out;
}

}  // namespace boltz
