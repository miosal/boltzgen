// Triangle multiplicative update (boltzgen.model.layers.triangular,
// TriangleMultiplicationOutgoing / Incoming) — the signature Pairformer pair op
// used by the design, folding and affinity models. Ports the pure-einsum
// reference path (the cuequivariance kernel is just a fused optimisation of the
// same math). Single example (B=1); x is [N, N, D] row-major, mask is [N, N].
//
// This is one of the custom kernels the Vulkan backend will need; here it is the
// validated scalar reference that the ggml/Vulkan version must match.
#pragma once

#include <vector>

namespace boltz {

struct TriMulWeights {
    int dim = 0;
    std::vector<float> norm_in_w, norm_in_b;   // [D]
    std::vector<float> p_in;                    // [2D, D] (no bias)
    std::vector<float> g_in;                    // [2D, D]
    std::vector<float> norm_out_w, norm_out_b;  // [D]
    std::vector<float> p_out;                   // [D, D]
    std::vector<float> g_out;                   // [D, D]
};

// direction: false = outgoing ("bikd,bjkd->bijd"), true = incoming
// ("bkid,bkjd->bijd"). Returns [N*N*D].
std::vector<float> triangle_multiplication(const std::vector<float>& x,
                                           const std::vector<float>& mask, int N, int D,
                                           const TriMulWeights& w, bool incoming);

}  // namespace boltz
