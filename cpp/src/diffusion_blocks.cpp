#include "boltz/diffusion_blocks.hpp"

#include <cmath>

namespace boltz {
namespace {

float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }
float siluf(float v) { return v / (1.0f + std::exp(-v)); }

// LayerNorm without affine.
std::vector<float> ln_noaffine(const std::vector<float>& x) {
    const int D = static_cast<int>(x.size());
    float mean = 0;
    for (float v : x) mean += v;
    mean /= D;
    float var = 0;
    for (float v : x) var += (v - mean) * (v - mean);
    var /= D;
    const float inv = 1.0f / std::sqrt(var + 1e-5f);
    std::vector<float> y(D);
    for (int d = 0; d < D; ++d) y[d] = (x[d] - mean) * inv;
    return y;
}

// LayerNorm with weight only (no bias).
std::vector<float> ln_weight(const std::vector<float>& x, const std::vector<float>& w) {
    auto y = ln_noaffine(x);
    for (size_t i = 0; i < y.size(); ++i) y[i] *= w[i];
    return y;
}

// y = W[out,in] @ x (+ b).
std::vector<float> lin(const std::vector<float>& x, const std::vector<float>& W, int out,
                       const std::vector<float>* b) {
    const int in = static_cast<int>(x.size());
    std::vector<float> y(out);
    for (int o = 0; o < out; ++o) {
        float s = b ? (*b)[o] : 0.0f;
        for (int i = 0; i < in; ++i) s += W[o * in + i] * x[i];
        y[o] = s;
    }
    return y;
}

}  // namespace

std::vector<float> adaln(const std::vector<float>& a, const std::vector<float>& s,
                         const AdaLNWeights& w) {
    const auto an = ln_noaffine(a);
    const auto sn = ln_weight(s, w.s_norm_w);
    const auto scale = lin(sn, w.s_scale_w, w.dim, &w.s_scale_b);
    const auto bias = lin(sn, w.s_bias_w, w.dim, nullptr);
    std::vector<float> out(w.dim);
    for (int d = 0; d < w.dim; ++d) out[d] = sigmoidf(scale[d]) * an[d] + bias[d];
    return out;
}

std::vector<float> conditioned_transition_block(const std::vector<float>& a,
                                                const std::vector<float>& s,
                                                const ConditionedTransitionWeights& w) {
    const auto a2 = adaln(a, s, w.adaln);
    // swish_gate: Linear -> [2*dim_inner], SwiGLU: first half=x, second=gates.
    const auto sg = lin(a2, w.swish_gate_w, 2 * w.dim_inner, nullptr);
    const auto atb = lin(a2, w.a_to_b_w, w.dim_inner, nullptr);
    std::vector<float> b(w.dim_inner);
    for (int i = 0; i < w.dim_inner; ++i)
        b[i] = (siluf(sg[w.dim_inner + i]) * sg[i]) * atb[i];
    const auto bta = lin(b, w.b_to_a_w, w.dim, nullptr);
    const auto op = lin(s, w.op_w, w.dim, &w.op_b);
    std::vector<float> out(w.dim);
    for (int d = 0; d < w.dim; ++d) out[d] = sigmoidf(op[d]) * bta[d];
    return out;
}

}  // namespace boltz
