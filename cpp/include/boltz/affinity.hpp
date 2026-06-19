// Affinity head (boltzgen.model.modules.affinity AffinityHeads.forward): pools
// the pair rep z over ligand/receptor cross-pairs, then ReLU MLPs predict a
// value, a score, and a binary logit. Single example (B=1).
#pragma once

#include <vector>

namespace boltz {

// A Linear with bias: w is [out,in], b is [out].
struct LinearB {
    std::vector<float> w, b;
    int in = 0, out = 0;
};

struct AffinityWeights {
    int token_z = 0, token_s = 0;  // token_s == input_token_s
    LinearB out_l1, out_l2;        // affinity_out_mlp: (tz->tz) relu (tz->ts) relu
    LinearB val_l1, val_l2, val_l3;  // to_affinity_pred_value: (ts->ts) relu (ts->ts) relu (ts->1)
    LinearB sco_l1, sco_l2, sco_l3;  // to_affinity_pred_score
    LinearB binary;                  // (1->1)
};

struct AffinityOutputs {
    float pred_value = 0;
    float pred_score = 0;
    float logits_binary = 0;
};

// z:[N*N*token_z]. lig_mask/rec_mask:[N] (1/0). Cross-pair pooling excludes the
// diagonal, matching the reference.
AffinityOutputs affinity_head(const std::vector<float>& z, const std::vector<float>& lig_mask,
                              const std::vector<float>& rec_mask, int N,
                              const AffinityWeights& w);

}  // namespace boltz
