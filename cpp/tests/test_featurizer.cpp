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

BOLTZ_TEST(single_sequence_msa) {
    std::vector<std::string> comps = {"ALA", "GLY", "VAL"};
    std::vector<int> seqs = {1, 2, 3};
    auto tf = token_features(comps, seqs);
    auto m = single_sequence_msa_features(tf.res_type, tf.n, tf.num_token_types);

    expect_eq_i(m.depth, 1, "single row");
    expect_eq_i(static_cast<long>(m.msa.size()), 3 * 33, "msa shape");
    // Profile equals the query one-hot.
    for (size_t i = 0; i < tf.res_type.size(); ++i) expect_eq_f(m.profile[i], tf.res_type[i], "profile==query");
    // No deletions, full mask.
    for (int i = 0; i < 3; ++i) {
        expect_eq_f(m.has_deletion[i], 0.0f, "no deletion");
        expect_eq_f(m.deletion_value[i], 0.0f, "zero deletion value");
        expect_eq_f(m.msa_mask[i], 1.0f, "mask 1");
    }
}

BOLTZ_TEST(msa_from_alignment_profile_and_deletions) {
    // 3 sequences, 2 columns, small token vocab T (use 33 for consistency).
    const int T = 33, n = 2, depth = 3;
    // Column 0 tokens: [2,2,5]; column 1: [7,7,7].
    std::vector<std::vector<int>> seqs = {{2, 7}, {2, 7}, {5, 7}};
    std::vector<std::vector<float>> dels = {{0, 1}, {2, 0}, {0, 0}};
    auto m = msa_features_from_alignment(seqs, dels, n, T);

    expect_eq_i(m.depth, 3, "depth");
    expect_eq_i(static_cast<long>(m.msa.size()), depth * n * T, "msa shape");
    // Profile col0: token 2 freq 2/3, token 5 freq 1/3.
    expect_eq_f(m.profile[0 * T + 2], 2.0f / 3.0f, "col0 tok2 freq");
    expect_eq_f(m.profile[0 * T + 5], 1.0f / 3.0f, "col0 tok5 freq");
    // Profile col1: token 7 freq 1.0.
    expect_eq_f(m.profile[1 * T + 7], 1.0f, "col1 tok7 freq");
    // Deletion mean: col0 = (0+2+0)/3, col1 = (1+0+0)/3.
    expect_eq_f(m.deletion_mean[0], 2.0f / 3.0f, "col0 del mean");
    expect_eq_f(m.deletion_mean[1], 1.0f / 3.0f, "col1 del mean");
    // has_deletion flags.
    expect_eq_f(m.has_deletion[0 * n + 1], 1.0f, "seq0 col1 has deletion");
    expect_eq_f(m.has_deletion[1 * n + 0], 1.0f, "seq1 col0 has deletion");
    expect_eq_f(m.has_deletion[2 * n + 0], 0.0f, "seq2 col0 no deletion");
}

int main() {
    std::printf("== test_featurizer ==\n");
    return boltztest::run_all();
}
