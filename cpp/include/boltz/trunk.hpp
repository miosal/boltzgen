// Pairformer trunk (stack of PairformerLayers) + distogram head. The shared
// trunk of the folding / design / affinity models. Single example (B=1).
#pragma once

#include "boltz/pairformer.hpp"

#include <vector>

namespace boltz {

// Apply a stack of Pairformer blocks in sequence.
PairformerState pairformer_trunk(const std::vector<float>& s, const std::vector<float>& z,
                                 const std::vector<float>& token_mask,
                                 const std::vector<float>& pair_mask, int N,
                                 const std::vector<PairformerWeights>& blocks);

// DistogramModule: z <- z + z^T (symmetrize over i,j), then Linear(token_z ->
// num_bins). lin_w is [num_bins, token_z], lin_b is [num_bins]. Returns
// [N*N*num_bins].
std::vector<float> distogram_head(const std::vector<float>& z, int N, int token_z,
                                  int num_bins, const std::vector<float>& lin_w,
                                  const std::vector<float>& lin_b);

}  // namespace boltz
