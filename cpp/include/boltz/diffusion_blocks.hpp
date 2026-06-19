// Diffusion-transformer conditioning blocks (boltzgen.model.modules.transformers):
// AdaLN (Algorithm 26) and ConditionedTransitionBlock (Algorithm 25). These
// condition the per-token diffusion features `a` on the single conditioning `s`.
// Operate on one token vector (apply per token in the transformer).
#pragma once

#include <vector>

namespace boltz {

struct AdaLNWeights {
    int dim = 0, dim_cond = 0;
    std::vector<float> s_norm_w;   // LayerNorm(dim_cond) weight (no bias)
    std::vector<float> s_scale_w;  // [dim, dim_cond] (+ bias)
    std::vector<float> s_scale_b;  // [dim]
    std::vector<float> s_bias_w;   // [dim, dim_cond] (no bias)
};

// a:[dim], s:[dim_cond] -> [dim].
// a_norm (no affine); s_norm (weight only); out = sigmoid(Ws·sn + bs)*a_n + Wb·sn.
std::vector<float> adaln(const std::vector<float>& a, const std::vector<float>& s,
                         const AdaLNWeights& w);

struct ConditionedTransitionWeights {
    int dim = 0, dim_cond = 0, dim_inner = 0;
    AdaLNWeights adaln;
    std::vector<float> swish_gate_w;  // [2*dim_inner, dim] (no bias)
    std::vector<float> a_to_b_w;      // [dim_inner, dim]
    std::vector<float> b_to_a_w;      // [dim, dim_inner]
    std::vector<float> op_w;          // [dim, dim_cond] (+ bias)
    std::vector<float> op_b;          // [dim]
};

// a:[dim], s:[dim_cond] -> [dim].
std::vector<float> conditioned_transition_block(const std::vector<float>& a,
                                                const std::vector<float>& s,
                                                const ConditionedTransitionWeights& w);

// FourierEmbedding (Algorithm 22): out[d] = cos(2*pi*(bias[d] + t*weight[d])).
// weight/bias are [dim]. Returns [dim] for a single time `t`.
std::vector<float> fourier_embedding(float t, const std::vector<float>& weight,
                                     const std::vector<float>& bias);

}  // namespace boltz
