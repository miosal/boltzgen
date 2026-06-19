#include "boltz/pairformer.hpp"

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

float silu(float v) { return v / (1.0f + std::exp(-v)); }

// One row of a SwiGLU Transition.
void transition_row(const float* x, const TransitionWeights& t, float* out) {
    std::vector<float> n(t.dim), h(t.hidden);
    layer_norm(x, t.dim, t.norm_w, t.norm_b, n.data());
    for (int o = 0; o < t.hidden; ++o) {
        float a = 0, b = 0;
        for (int i = 0; i < t.dim; ++i) {
            a += t.fc1[o * t.dim + i] * n[i];
            b += t.fc2[o * t.dim + i] * n[i];
        }
        h[o] = silu(a) * b;
    }
    for (int o = 0; o < t.out; ++o) {
        float s = 0;
        for (int i = 0; i < t.hidden; ++i) s += t.fc3[o * t.hidden + i] * h[i];
        out[o] = s;
    }
}

void add_into(std::vector<float>& dst, const std::vector<float>& src) {
    for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
}

}  // namespace

PairformerState pairformer_block(const std::vector<float>& s_in,
                                 const std::vector<float>& z_in,
                                 const std::vector<float>& token_mask,
                                 const std::vector<float>& pair_mask, int N,
                                 const PairformerWeights& w) {
    const int ts = w.token_s, tz = w.token_z;
    std::vector<float> s = s_in, z = z_in;

    // Pair stack (residual updates).
    add_into(z, triangle_multiplication(z, pair_mask, N, tz, w.tri_mul_out, /*incoming=*/false));
    add_into(z, triangle_multiplication(z, pair_mask, N, tz, w.tri_mul_in, /*incoming=*/true));
    add_into(z, triangle_attention(z, pair_mask, N, w.tri_att_start, /*starting=*/true));
    add_into(z, triangle_attention(z, pair_mask, N, w.tri_att_end, /*starting=*/false));
    {
        std::vector<float> zt((size_t)N * N * tz);
        for (int p = 0; p < N * N; ++p) transition_row(&z[p * tz], w.transition_z, &zt[p * tz]);
        add_into(z, zt);
    }

    // Sequence stack.
    std::vector<float> s_normed((size_t)N * ts);
    for (int i = 0; i < N; ++i)
        layer_norm(&s[i * ts], ts, w.pre_norm_s_w, w.pre_norm_s_b, &s_normed[i * ts]);

    // Attention key mask: mask key j by token_mask[j] (broadcast over queries).
    std::vector<float> mask2d((size_t)N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) mask2d[i * N + j] = token_mask[j];

    add_into(s, attention_pair_bias(s_normed, z, mask2d, N, w.attention));

    {
        std::vector<float> st((size_t)N * ts);
        for (int i = 0; i < N; ++i) transition_row(&s[i * ts], w.transition_s, &st[i * ts]);
        add_into(s, st);
    }

    return {std::move(s), std::move(z)};
}

}  // namespace boltz
