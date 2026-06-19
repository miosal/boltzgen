#include "boltz/rel_pos.hpp"

#include <algorithm>

namespace boltz {
namespace {

int clip(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

}  // namespace

std::vector<float> relative_position_features(const RelPosInputs& in, int N, int r_max,
                                              int s_max) {
    const int n_res = 2 * r_max + 2;    // a_rel_pos / a_rel_token width
    const int n_chain = 2 * s_max + 2;  // a_rel_chain width
    const int dim = rel_pos_feature_dim(r_max, s_max);

    std::vector<float> out((size_t)N * N * dim, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float* f = &out[((size_t)i * N + j) * dim];
            const bool same_chain = in.feature_asym_id[i] == in.feature_asym_id[j];
            const bool same_residue = in.feature_residue_index[i] == in.feature_residue_index[j];
            const bool same_entity = in.entity_id[i] == in.entity_id[j];
            int off = 0;

            // a_rel_pos
            int d_res = same_chain ? clip(in.feature_residue_index[i] - in.feature_residue_index[j] + r_max, 0, 2 * r_max)
                                   : (2 * r_max + 1);
            f[off + d_res] = 1.0f;
            off += n_res;

            // a_rel_token
            int d_tok = (same_chain && same_residue)
                            ? clip(in.token_index[i] - in.token_index[j] + r_max, 0, 2 * r_max)
                            : (2 * r_max + 1);
            f[off + d_tok] = 1.0f;
            off += n_res;

            // a_rel_chain
            int d_chain = same_entity ? clip(in.sym_id[i] - in.sym_id[j] + s_max, 0, 2 * s_max)
                                      : (2 * s_max + 1);
            f[off + d_chain] = 1.0f;
            off += n_chain;

            // same_entity flag
            f[off] = same_entity ? 1.0f : 0.0f;
        }
    }
    return out;
}

std::vector<float> relative_position_encode(const RelPosInputs& in, int N, int token_z,
                                            const std::vector<float>& w, int r_max, int s_max) {
    const int dim = rel_pos_feature_dim(r_max, s_max);
    const std::vector<float> feats = relative_position_features(in, N, r_max, s_max);
    std::vector<float> out((size_t)N * N * token_z);
    for (int p = 0; p < N * N; ++p) {
        const float* f = &feats[(size_t)p * dim];
        for (int o = 0; o < token_z; ++o) {
            float s = 0;
            for (int c = 0; c < dim; ++c) s += w[o * dim + c] * f[c];
            out[(size_t)p * token_z + o] = s;
        }
    }
    return out;
}

}  // namespace boltz
