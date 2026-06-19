// Tests for standard-residue atom featurization (baked heavy-atom topology).
#include "boltz/atom_featurizer.hpp"
#include "test_framework.hpp"

#include <vector>

using namespace boltz;
using namespace boltztest;

BOLTZ_TEST(standard_atoms_counts) {
    expect_eq_i(static_cast<long>(standard_residue_atoms("GLY").size()), 4, "GLY backbone only");
    expect_eq_i(static_cast<long>(standard_residue_atoms("ALA").size()), 5, "ALA + CB");
    expect_eq_i(static_cast<long>(standard_residue_atoms("TRP").size()), 14, "TRP largest");
    expect_true(standard_residue_atoms("LIG").empty(), "non-standard empty");
}

BOLTZ_TEST(element_inference) {
    expect_eq_i(element_of_atom("N"), 7, "N");
    expect_eq_i(element_of_atom("CA"), 6, "C");
    expect_eq_i(element_of_atom("OD1"), 8, "O");
    expect_eq_i(element_of_atom("SG"), 16, "S");
    expect_eq_i(element_of_atom("NH1"), 7, "N (multi-char)");
}

BOLTZ_TEST(atom_features_layout) {
    // GLY(4) + ALA(5) + unknown(1) = 10 atoms.
    std::vector<std::string> comps = {"GLY", "ALA", "LIG"};
    auto f = atom_features(comps);
    expect_eq_i(f.n_atoms, 10, "total atoms");

    // Token starts / counts.
    expect_eq_i(f.token_atom_start[0], 0, "GLY start");
    expect_eq_i(f.token_atom_count[0], 4, "GLY count");
    expect_eq_i(f.token_atom_start[1], 4, "ALA start");
    expect_eq_i(f.token_atom_count[1], 5, "ALA count");
    expect_eq_i(f.token_atom_start[2], 9, "LIG start");
    expect_eq_i(f.token_atom_count[2], 1, "LIG placeholder");

    // Representative atom is CA: GLY CA at index 1, ALA CA at index 5.
    expect_eq_i(f.token_to_rep_atom[0], 1, "GLY rep = CA");
    expect_eq_i(f.token_to_rep_atom[1], 5, "ALA rep = CA");
    expect_true(f.atom_name[f.token_to_rep_atom[0]] == "CA", "rep is CA");

    // atom_to_token mapping.
    for (int a = 0; a < 4; ++a) expect_eq_i(f.atom_to_token[a], 0, "GLY atoms -> token 0");
    for (int a = 4; a < 9; ++a) expect_eq_i(f.atom_to_token[a], 1, "ALA atoms -> token 1");
    expect_eq_i(f.atom_to_token[9], 2, "LIG atom -> token 2");

    // Backbone mask: first 4 of each residue's N/CA/C/O are backbone; CB is not.
    expect_eq_f(f.backbone_mask[4], 1.0f, "ALA N backbone");   // N
    expect_eq_f(f.backbone_mask[8], 0.0f, "ALA CB sidechain"); // CB
}

int main() {
    std::printf("== test_atom_featurizer ==\n");
    return boltztest::run_all();
}
