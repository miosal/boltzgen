// Token-level part of boltzgen.model.modules.trunk.InputEmbedder.forward — the
// initial single representation s, minus the atom-derived term `a` (which comes
// from the separately-built atom encoder):
//
//   s = res_type_encoding(res_type)
//     + msa_profile_encoding(cat[profile, deletion_mean])
//     + [optional] conditioning embeddings (mol_type, design_mask, binding_type)
//
// res/profile encoders are bias-free Linears (LinearNoBias); conditioning inits
// are Embeddings. Pure linear algebra / lookups — no weights-from-HF required to
// validate the wiring.
#pragma once

#include <vector>

namespace boltz {

struct InputEmbedderWeights {
    int token_s = 0;
    int num_tokens = 0;  // T (res_type / profile one-hot width)

    std::vector<float> res_type_w;  // [token_s * T]          (LinearNoBias)
    std::vector<float> profile_w;   // [token_s * (T + 1)]    (LinearNoBias; +deletion_mean)

    // Optional conditioning embeddings (Embedding tables [K * token_s]).
    bool add_mol_type = false;
    std::vector<float> mol_type_emb;     // [4 * token_s]
    bool add_design_mask = false;
    std::vector<float> design_mask_emb;  // [2 * token_s]
    bool add_binding = false;
    std::vector<float> binding_emb;      // [3 * token_s]
};

// Inputs are per-token: res_type/profile are [N*T] one-hot/float; deletion_mean
// is [N]; mol_type/design_mask/binding_type are [N] integer category ids (used
// only when the corresponding add_* flag is set). Returns s as [N*token_s].
std::vector<float> token_input_embedding(const InputEmbedderWeights& w, int N,
                                         const std::vector<float>& res_type,
                                         const std::vector<float>& profile,
                                         const std::vector<float>& deletion_mean,
                                         const std::vector<int>& mol_type,
                                         const std::vector<int>& design_mask,
                                         const std::vector<int>& binding_type);

}  // namespace boltz
