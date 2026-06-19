// DiffusionTransformer (boltzgen.model.modules.transformers, Algorithm 23) — the
// token-level transformer inside the diffusion module. Each layer:
//   b = AdaLN(a, s); b = PairBiasAttn(b, bias, mask) [compute_pair_bias=false];
//   b = sigmoid(Wop s + bop) * b; a = a + b; a = a + CondTransition(a, s).
// Single example (B=1).
#pragma once

#include "boltz/attention_pair_bias.hpp"
#include "boltz/diffusion_blocks.hpp"

#include <vector>

namespace boltz {

struct DiffusionTransformerLayerWeights {
    int dim = 0, dim_cond = 0, num_heads = 0;
    AdaLNWeights adaln;
    AttnPairBiasWeights attn;  // compute_pair_bias = false; per-head bias supplied
    std::vector<float> op_w;   // [dim, dim_cond]
    std::vector<float> op_b;   // [dim]
    ConditionedTransitionWeights transition;
};

// a:[N*dim], s:[N*dim_cond], bias:[N*N*num_heads] (per-head), token_mask:[N].
std::vector<float> diffusion_transformer_layer(const std::vector<float>& a,
                                               const std::vector<float>& s,
                                               const std::vector<float>& bias,
                                               const std::vector<float>& token_mask, int N,
                                               const DiffusionTransformerLayerWeights& w);

// Stack of layers; `biases[l]` is layer l's per-head bias [N*N*num_heads].
std::vector<float> diffusion_transformer(const std::vector<float>& a,
                                         const std::vector<float>& s,
                                         const std::vector<std::vector<float>>& biases,
                                         const std::vector<float>& token_mask, int N,
                                         const std::vector<DiffusionTransformerLayerWeights>& layers);

}  // namespace boltz
