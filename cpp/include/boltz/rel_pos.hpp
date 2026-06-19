// RelativePositionEncoder (boltzgen.model.modules.encoders, Algorithm 3) — a
// featurizer component used by the trunk input embedding. Builds per-pair
// relative-position one-hot features from token/residue/chain/entity indices,
// then projects to token_z. Pure index math (no chemistry / no weights for the
// feature part). Non-cyclic path. Single example (B=1).
#pragma once

#include <vector>

namespace boltz {

// Per-token integer features (length N each).
struct RelPosInputs {
    std::vector<int> feature_asym_id;       // chain id
    std::vector<int> feature_residue_index;  // residue index
    std::vector<int> entity_id;
    std::vector<int> token_index;
    std::vector<int> sym_id;                 // symmetry copy id
};

// Feature width for given r_max/s_max: 4*(r_max+1) + 2*(s_max+1) + 1.
inline int rel_pos_feature_dim(int r_max, int s_max) {
    return 4 * (r_max + 1) + 2 * (s_max + 1) + 1;
}

// Build the concatenated relative-position features: [N*N*feature_dim], in the
// order [a_rel_pos | a_rel_token | a_rel_chain | same_entity].
std::vector<float> relative_position_features(const RelPosInputs& in, int N, int r_max = 32,
                                              int s_max = 2);

// Project the features to token_z via a bias-free Linear (w is [token_z, dim]).
// Returns [N*N*token_z].
std::vector<float> relative_position_encode(const RelPosInputs& in, int N, int token_z,
                                            const std::vector<float>& w, int r_max = 32,
                                            int s_max = 2);

}  // namespace boltz
