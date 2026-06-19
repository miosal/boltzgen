// OuterProductMean (boltzgen.model.layers.outer_product_mean) — aggregates the
// MSA representation into the pair representation in the trunk. Single example
// (B=1). m is [S, N, c_in], mask is [S, N]; output is [N, N, c_out].
//
// z[i,j] = proj_o( ( sum_s a[s,i] (outer) b[s,j] ) / max(1, sum_s mask[s,i]mask[s,j]) )
// where a = proj_a(norm(m))*mask, b = proj_b(norm(m))*mask.
#pragma once

#include <vector>

namespace boltz {

struct OuterProductMeanWeights {
    int c_in = 0, c_hidden = 0, c_out = 0;
    std::vector<float> norm_w, norm_b;  // [c_in]
    std::vector<float> proj_a;          // [c_hidden, c_in] (no bias)
    std::vector<float> proj_b;          // [c_hidden, c_in]
    std::vector<float> proj_o_w;        // [c_out, c_hidden*c_hidden]
    std::vector<float> proj_o_b;        // [c_out]
};

// m: [S*N*c_in], mask: [S*N]. Returns [N*N*c_out].
std::vector<float> outer_product_mean(const std::vector<float>& m,
                                      const std::vector<float>& mask, int S, int N,
                                      const OuterProductMeanWeights& w);

}  // namespace boltz
