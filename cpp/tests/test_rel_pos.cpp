// Tests for the RelativePositionEncoder feature construction (Algorithm 3).
#include "boltz/rel_pos.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

BOLTZ_TEST(rel_pos_feature_dim_value) {
    expect_eq_i(rel_pos_feature_dim(32, 2), 4 * 33 + 2 * 3 + 1, "default dim 139");
    expect_eq_i(rel_pos_feature_dim(2, 1), 4 * 3 + 2 * 2 + 1, "small dim");
}

BOLTZ_TEST(rel_pos_onehot_structure) {
    // r_max=2, s_max=1: n_res=6, n_chain=4, dim = 4*3+2*2+1 = 17.
    const int r = 2, s = 1, N = 3;
    RelPosInputs in;
    // tokens 0,1 same chain/entity/residue-distinct; token 2 different chain+entity.
    in.feature_asym_id      = {0, 0, 1};
    in.feature_residue_index = {5, 6, 5};
    in.entity_id            = {0, 0, 1};
    in.token_index          = {0, 1, 2};
    in.sym_id               = {0, 0, 0};

    auto f = relative_position_features(in, N, r, s);
    const int dim = rel_pos_feature_dim(r, s);
    expect_eq_i(static_cast<long>(f.size()), N * N * dim, "shape");

    const int n_res = 2 * r + 2;   // 6
    const int n_chain = 2 * s + 2; // 4
    auto blk = [&](int i, int j) { return &f[(i * N + j) * dim]; };

    // Each of the three one-hot blocks has exactly one 1; same_entity flag 0/1.
    auto count_block = [](const float* p, int w) { int c = 0; for (int k = 0; k < w; ++k) c += (p[k] == 1.0f); return c; };

    // Pair (0,1): same chain, same entity, residue diff = 5-6 = -1 -> clip(-1+2,0,4)=1.
    const float* f01 = blk(0, 1);
    expect_eq_i(count_block(f01, n_res), 1, "rel_pos one-hot");
    expect_eq_f(f01[1], 1.0f, "rel_pos index = 1");
    // rel_token: same chain but different residue -> out-of-window class 2r+1 = 5.
    expect_eq_f(f01[n_res + (2 * r + 1)], 1.0f, "rel_token out-of-residue");
    // same_entity flag (last element) = 1.
    expect_eq_f(f01[dim - 1], 1.0f, "same entity");

    // Pair (0,2): different chain AND different entity.
    const float* f02 = blk(0, 2);
    // rel_pos: not same chain -> class 2r+1 = 5.
    expect_eq_f(f02[2 * r + 1], 1.0f, "rel_pos diff chain");
    // rel_chain: not same entity -> class 2s+1 = 3.
    expect_eq_f(f02[2 * n_res + (2 * s + 1)], 1.0f, "rel_chain diff entity");
    // same_entity flag = 0.
    expect_eq_f(f02[dim - 1], 0.0f, "diff entity flag");

    // Diagonal (i,i): same everything, residue/token diff 0 -> class r_max.
    const float* f00 = blk(0, 0);
    expect_eq_f(f00[r], 1.0f, "rel_pos self = r_max");
    expect_eq_f(f00[n_res + r], 1.0f, "rel_token self = r_max");
    expect_eq_f(f00[2 * n_res + s], 1.0f, "rel_chain self = s_max");
    expect_eq_f(f00[dim - 1], 1.0f, "self same entity");
}

BOLTZ_TEST(rel_pos_encode_projects) {
    const int r = 2, s = 1, N = 2, token_z = 3;
    RelPosInputs in;
    in.feature_asym_id = {0, 0}; in.feature_residue_index = {1, 2};
    in.entity_id = {0, 0}; in.token_index = {0, 1}; in.sym_id = {0, 0};
    const int dim = rel_pos_feature_dim(r, s);
    std::vector<float> w(token_z * dim);
    for (int i = 0; i < token_z * dim; ++i) w[i] = 0.01f * std::sin(i);

    auto z = relative_position_encode(in, N, token_z, w, r, s);
    expect_eq_i(static_cast<long>(z.size()), N * N * token_z, "encoded shape");
    // Spot-check (0,0) against direct projection of its feature row.
    auto f = relative_position_features(in, N, r, s);
    for (int o = 0; o < token_z; ++o) {
        float ref = 0;
        for (int c = 0; c < dim; ++c) ref += w[o * dim + c] * f[c];
        expect_eq_f(z[o], ref, "projection");
    }
}

int main() {
    std::printf("== test_rel_pos ==\n");
    return boltztest::run_all();
}
