#include "boltz/input_embedder.hpp"

namespace boltz {

std::vector<float> token_input_embedding(const InputEmbedderWeights& w, int N,
                                         const std::vector<float>& res_type,
                                         const std::vector<float>& profile,
                                         const std::vector<float>& deletion_mean,
                                         const std::vector<int>& mol_type,
                                         const std::vector<int>& design_mask,
                                         const std::vector<int>& binding_type) {
    const int ts = w.token_s;
    const int T = w.num_tokens;
    std::vector<float> s((size_t)N * ts, 0.0f);

    for (int i = 0; i < N; ++i) {
        float* si = &s[(size_t)i * ts];

        // res_type_encoding: LinearNoBias [ts, T] applied to res_type[i].
        for (int o = 0; o < ts; ++o) {
            float acc = 0.0f;
            for (int c = 0; c < T; ++c) acc += w.res_type_w[o * T + c] * res_type[(size_t)i * T + c];
            si[o] += acc;
        }

        // msa_profile_encoding: LinearNoBias [ts, T+1] applied to [profile[i], deletion_mean[i]].
        for (int o = 0; o < ts; ++o) {
            float acc = 0.0f;
            for (int c = 0; c < T; ++c) acc += w.profile_w[o * (T + 1) + c] * profile[(size_t)i * T + c];
            acc += w.profile_w[o * (T + 1) + T] * deletion_mean[i];
            si[o] += acc;
        }

        // Optional conditioning embeddings (lookup add).
        if (w.add_mol_type) {
            const float* e = &w.mol_type_emb[(size_t)mol_type[i] * ts];
            for (int o = 0; o < ts; ++o) si[o] += e[o];
        }
        if (w.add_design_mask) {
            const float* e = &w.design_mask_emb[(size_t)design_mask[i] * ts];
            for (int o = 0; o < ts; ++o) si[o] += e[o];
        }
        if (w.add_binding) {
            const float* e = &w.binding_emb[(size_t)binding_type[i] * ts];
            for (int o = 0; o < ts; ++o) si[o] += e[o];
        }
    }
    return s;
}

}  // namespace boltz
