// Triangle self-attention (boltzgen.model.layers.triangular_attention,
// TriangleAttention starting/ending node — AF2 Algorithm 13). Single example
// (B=1). x is [N, N, c_in]; output is [N, N, c_in].
//
// Semantics (derived from the reference broadcasting, starting node):
//   xn = LayerNorm(x)
//   per row i, attention over key positions k within that row:
//     logit[i,h,q,k] = scale * (Wq xn[i,q]) . (Wk xn[i,k])
//                      + triBias[h,q,k] + inf*(mask[i,k]-1)
//   where triBias[h,q,k] = (W_tri xn[q,k])[h]  (shared across rows i),
//   scale = 1/sqrt(head_dim). Output gated by sigmoid(Wg xn[i,q]), then Wo.
// Ending node = same op on the transpose of x.
//
// NOTE: this is the validated *wiring*; the bias-broadcasting interpretation
// still needs a golden-tensor parity check against the reference model.
#pragma once

#include <vector>

namespace boltz {

struct TriAttnWeights {
    int c_in = 0;
    int num_heads = 0;
    int head_dim = 0;  // c_hidden (per-head)
    std::vector<float> norm_w, norm_b;  // [c_in]
    std::vector<float> tri_w;           // [num_heads, c_in] (no bias)
    std::vector<float> q_w, k_w, v_w;   // [num_heads*head_dim, c_in] (no bias)
    std::vector<float> g_w;             // [num_heads*head_dim, c_in]
    std::vector<float> o_w;             // [c_in, num_heads*head_dim]
    float inf = 1e9f;
};

// x: [N*N*c_in], mask: [N*N]. starting=false => ending node. Returns [N*N*c_in].
std::vector<float> triangle_attention(const std::vector<float>& x,
                                      const std::vector<float>& mask, int N,
                                      const TriAttnWeights& w, bool starting);

}  // namespace boltz
