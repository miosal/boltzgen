#include "boltz/trunk.hpp"

namespace boltz {

PairformerState pairformer_trunk(const std::vector<float>& s, const std::vector<float>& z,
                                 const std::vector<float>& token_mask,
                                 const std::vector<float>& pair_mask, int N,
                                 const std::vector<PairformerWeights>& blocks) {
    PairformerState st{s, z};
    for (const PairformerWeights& w : blocks)
        st = pairformer_block(st.s, st.z, token_mask, pair_mask, N, w);
    return st;
}

std::vector<float> distogram_head(const std::vector<float>& z, int N, int token_z,
                                  int num_bins, const std::vector<float>& lin_w,
                                  const std::vector<float>& lin_b) {
    // Symmetrize: zs[i,j] = z[i,j] + z[j,i].
    std::vector<float> zs((size_t)N * N * token_z);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int c = 0; c < token_z; ++c)
                zs[(i * N + j) * token_z + c] =
                    z[(i * N + j) * token_z + c] + z[(j * N + i) * token_z + c];

    std::vector<float> out((size_t)N * N * num_bins);
    for (int p = 0; p < N * N; ++p) {
        const float* in = &zs[p * token_z];
        for (int b = 0; b < num_bins; ++b) {
            float s = lin_b[b];
            for (int c = 0; c < token_z; ++c) s += lin_w[b * token_z + c] * in[c];
            out[p * num_bins + b] = s;
        }
    }
    return out;
}

}  // namespace boltz
