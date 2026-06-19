// Equivalence test for the token-level InputEmbedder against an independent
// scalar reference.
#include "boltz/input_embedder.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

BOLTZ_TEST(token_input_embedding_matches_reference) {
    const int N = 3, ts = 4, T = 5;
    InputEmbedderWeights w;
    w.token_s = ts; w.num_tokens = T;
    w.res_type_w.resize(ts * T);
    w.profile_w.resize(ts * (T + 1));
    for (int i = 0; i < ts * T; ++i) w.res_type_w[i] = 0.1f * std::sin(1.0f + i);
    for (int i = 0; i < ts * (T + 1); ++i) w.profile_w[i] = 0.1f * std::cos(2.0f + i);
    w.add_mol_type = true; w.mol_type_emb.resize(4 * ts);
    for (int i = 0; i < 4 * ts; ++i) w.mol_type_emb[i] = 0.01f * (i + 1);
    w.add_design_mask = true; w.design_mask_emb.resize(2 * ts);
    for (int i = 0; i < 2 * ts; ++i) w.design_mask_emb[i] = -0.02f * (i + 1);
    w.add_binding = true; w.binding_emb.resize(3 * ts);
    for (int i = 0; i < 3 * ts; ++i) w.binding_emb[i] = 0.03f * (i + 1);

    // Inputs.
    std::vector<float> res_type(N * T, 0.0f), profile(N * T, 0.0f);
    std::vector<float> deletion_mean = {0.0f, 0.5f, 1.0f};
    std::vector<int> mol_type = {0, 2, 3}, design_mask = {0, 1, 1}, binding_type = {0, 1, 2};
    for (int i = 0; i < N; ++i) {
        res_type[i * T + (i % T)] = 1.0f;                 // one-hot
        for (int c = 0; c < T; ++c) profile[i * T + c] = 0.1f * (c + 1) * (i + 1);  // soft profile
    }

    auto s = token_input_embedding(w, N, res_type, profile, deletion_mean, mol_type,
                                   design_mask, binding_type);

    // Independent scalar reference.
    std::vector<float> ref((size_t)N * ts, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int o = 0; o < ts; ++o) {
            float acc = 0;
            for (int c = 0; c < T; ++c) acc += w.res_type_w[o * T + c] * res_type[i * T + c];
            for (int c = 0; c < T; ++c) acc += w.profile_w[o * (T + 1) + c] * profile[i * T + c];
            acc += w.profile_w[o * (T + 1) + T] * deletion_mean[i];
            acc += w.mol_type_emb[mol_type[i] * ts + o];
            acc += w.design_mask_emb[design_mask[i] * ts + o];
            acc += w.binding_emb[binding_type[i] * ts + o];
            ref[i * ts + o] = acc;
        }
    }
    for (int i = 0; i < N * ts; ++i) expect_eq_f(s[i], ref[i], "s");
}

BOLTZ_TEST(token_input_embedding_conditioning_optional) {
    // With all add_* flags off, only res_type + profile contribute.
    const int N = 1, ts = 2, T = 2;
    InputEmbedderWeights w;
    w.token_s = ts; w.num_tokens = T;
    w.res_type_w = {1, 0, 0, 1};           // identity-ish [ts,T]
    w.profile_w = {0, 0, 0, 0, 0, 0};      // zero [ts, T+1]
    std::vector<float> res_type = {1, 0};  // token 0
    std::vector<float> profile = {0, 0};
    std::vector<float> del = {0};
    auto s = token_input_embedding(w, N, res_type, profile, del, {}, {}, {});
    expect_eq_f(s[0], 1.0f, "res_type row0");
    expect_eq_f(s[1], 0.0f, "res_type row1");
}

int main() {
    std::printf("== test_input_embedder ==\n");
    return boltztest::run_all();
}
