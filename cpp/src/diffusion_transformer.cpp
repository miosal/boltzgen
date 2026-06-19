#include "boltz/diffusion_transformer.hpp"

#include <cmath>

namespace boltz {
namespace {

float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

std::vector<float> slice(const std::vector<float>& v, int i, int d) {
    return std::vector<float>(v.begin() + (size_t)i * d, v.begin() + (size_t)(i + 1) * d);
}

}  // namespace

std::vector<float> diffusion_transformer_layer(const std::vector<float>& a,
                                               const std::vector<float>& s,
                                               const std::vector<float>& bias,
                                               const std::vector<float>& token_mask, int N,
                                               const DiffusionTransformerLayerWeights& w) {
    const int dim = w.dim, dc = w.dim_cond;

    // b = AdaLN(a, s) per token.
    std::vector<float> b((size_t)N * dim);
    for (int i = 0; i < N; ++i) {
        auto bi = adaln(slice(a, i, dim), slice(s, i, dc), w.adaln);
        for (int d = 0; d < dim; ++d) b[i * dim + d] = bi[d];
    }

    // Self-attention with the supplied per-head bias.
    std::vector<float> mask2d((size_t)N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) mask2d[i * N + j] = token_mask[j];
    auto attn = attention_pair_bias(b, bias, mask2d, N, w.attn);

    // Output gate sigmoid(Wop s + bop), residual, conditioned transition.
    std::vector<float> out = a;
    for (int i = 0; i < N; ++i) {
        std::vector<float> op(dim);
        for (int d = 0; d < dim; ++d) {
            float acc = w.op_b[d];
            for (int c = 0; c < dc; ++c) acc += w.op_w[d * dc + c] * s[i * dc + c];
            op[d] = sigmoidf(acc);
        }
        for (int d = 0; d < dim; ++d) out[i * dim + d] += op[d] * attn[i * dim + d];
    }
    for (int i = 0; i < N; ++i) {
        auto t = conditioned_transition_block(slice(out, i, dim), slice(s, i, dc), w.transition);
        for (int d = 0; d < dim; ++d) out[i * dim + d] += t[d];
    }
    return out;
}

std::vector<float> diffusion_transformer(const std::vector<float>& a_in,
                                         const std::vector<float>& s,
                                         const std::vector<std::vector<float>>& biases,
                                         const std::vector<float>& token_mask, int N,
                                         const std::vector<DiffusionTransformerLayerWeights>& layers) {
    std::vector<float> a = a_in;
    for (size_t l = 0; l < layers.size(); ++l)
        a = diffusion_transformer_layer(a, s, biases[l], token_mask, N, layers[l]);
    return a;
}

}  // namespace boltz
