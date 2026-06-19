// Tests for hand-rolled featurization helpers. Expected values derived directly
// from the GaussianSmearing formula in boltzgen.model.modules.inverse_fold.
#include "boltz/featurizer.hpp"
#include "boltz/const.hpp"
#include "test_framework.hpp"

#include <cmath>

using namespace boltz;
using namespace boltztest;

BOLTZ_TEST(gaussian_smearing_shape) {
    auto m = gaussian_smearing({0.0f, 1.0f, 2.0f}, 0.0f, 5.0f, 6);
    expect_eq_i(m.rows, 3, "rows = n_dist");
    expect_eq_i(m.cols, 6, "cols = n_gaussians");
}

BOLTZ_TEST(gaussian_smearing_values) {
    // offsets = [0,1,2,3,4,5], delta = 1, coeff = -0.5.
    auto m = gaussian_smearing({2.0f}, 0.0f, 5.0f, 6);
    const float coeff = -0.5f;
    for (int k = 0; k < 6; ++k) {
        float d = 2.0f - static_cast<float>(k);
        expect_eq_f(m.at(0, k), std::exp(coeff * d * d), "smear value");
    }
    // Peak (=1.0) exactly at the offset equal to the distance.
    expect_eq_f(m.at(0, 2), 1.0f, "peak at offset==dist");
}

BOLTZ_TEST(gaussian_smearing_16_bins_range20) {
    // Inverse-folding default: 16 gaussians over 0..20 A.
    auto m = gaussian_smearing({0.0f}, 0.0f, 20.0f, 16);
    expect_eq_i(m.cols, 16, "16 bins");
    expect_eq_f(m.at(0, 0), 1.0f, "dist 0 peaks at first offset (0)");
}

BOLTZ_TEST(token_features_onehot_and_moltype) {
    std::vector<std::string> comps = {"ALA", "GLY", "DA", "A", "LIG"};
    std::vector<int> seqs = {3, 4, 10, 20, 1};
    auto f = token_features(comps, seqs);

    expect_eq_i(f.n, 5, "n");
    expect_eq_i(f.num_token_types, 33, "33 token types");
    expect_eq_i(static_cast<long>(f.res_type.size()), 5 * 33, "res_type shape");

    // One-hot: exactly one 1 per row, at the token id.
    for (int i = 0; i < 5; ++i) {
        int ones = 0, where = -1;
        for (int t = 0; t < 33; ++t) if (f.res_type[i * 33 + t] == 1.0f) { ones++; where = t; }
        expect_eq_i(ones, 1, "one-hot single");
        expect_eq_i(where, token_id(comps[i]), "one-hot position");
    }
    // ALA at token id 2; "LIG" -> UNK id.
    expect_eq_i(token_id("ALA"), 2, "ALA id");
    expect_eq_i(token_id("LIG"), token_id("UNK"), "unknown -> UNK");

    // mol types: protein, protein, DNA, RNA, nonpolymer.
    expect_eq_i(f.mol_type[0], MOL_PROTEIN, "ALA protein");
    expect_eq_i(f.mol_type[2], MOL_DNA, "DA dna");
    expect_eq_i(f.mol_type[3], MOL_RNA, "A rna");
    expect_eq_i(f.mol_type[4], MOL_NONPOLYMER, "LIG nonpolymer");

    expect_eq_i(f.token_index[2], 2, "token index");
    expect_eq_i(f.residue_index[2], 10, "residue index from seq");
}

int main() {
    std::printf("== test_featurizer ==\n");
    return boltztest::run_all();
}
