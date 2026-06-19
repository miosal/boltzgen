// Tests for the heavy-atom SMILES parser.
#include "boltz/smiles.hpp"
#include "test_framework.hpp"

using namespace boltz;
using namespace boltztest;

static int count_order(const SmilesMol& m, int order) {
    int n = 0; for (auto& b : m.bonds) if (b.order == order && !b.aromatic) ++n; return n;
}
static int count_aromatic_bonds(const SmilesMol& m) {
    int n = 0; for (auto& b : m.bonds) if (b.aromatic) ++n; return n;
}

BOLTZ_TEST(smiles_single_atom) {
    auto m = parse_smiles("C");
    expect_eq_i(static_cast<long>(m.atoms.size()), 1, "1 atom");
    expect_eq_i(m.atoms[0].element, 6, "carbon");
    expect_eq_i(static_cast<long>(m.bonds.size()), 0, "no bonds");
}

BOLTZ_TEST(smiles_chain_and_double_bond) {
    auto m = parse_smiles("CC=O");  // ethanal heavy atoms: C-C=O
    expect_eq_i(static_cast<long>(m.atoms.size()), 3, "3 atoms");
    expect_eq_i(m.atoms[2].element, 8, "oxygen");
    expect_eq_i(static_cast<long>(m.bonds.size()), 2, "2 bonds");
    expect_eq_i(count_order(m, 1), 1, "one single");
    expect_eq_i(count_order(m, 2), 1, "one double");
}

BOLTZ_TEST(smiles_two_letter_elements) {
    auto m = parse_smiles("ClBr");  // not a real molecule, tests Cl/Br tokenizing
    expect_eq_i(static_cast<long>(m.atoms.size()), 2, "2 atoms");
    expect_eq_i(m.atoms[0].element, 17, "Cl");
    expect_eq_i(m.atoms[1].element, 35, "Br");
}

BOLTZ_TEST(smiles_benzene_ring_aromatic) {
    auto m = parse_smiles("c1ccccc1");
    expect_eq_i(static_cast<long>(m.atoms.size()), 6, "6 carbons");
    for (auto& a : m.atoms) { expect_true(a.aromatic, "aromatic"); expect_eq_i(a.element, 6, "C"); }
    // 5 chain bonds + 1 ring-closure bond = 6 aromatic bonds.
    expect_eq_i(static_cast<long>(m.bonds.size()), 6, "6 bonds");
    expect_eq_i(count_aromatic_bonds(m), 6, "all aromatic");
}

BOLTZ_TEST(smiles_branch) {
    auto m = parse_smiles("C(C)C");  // central carbon, two branches
    expect_eq_i(static_cast<long>(m.atoms.size()), 3, "3 atoms");
    expect_eq_i(static_cast<long>(m.bonds.size()), 2, "2 bonds");
    // atom 0 bonds to atom 1 and atom 2.
    int deg0 = 0; for (auto& b : m.bonds) if (b.i == 0 || b.j == 0) ++deg0;
    expect_eq_i(deg0, 2, "central atom degree 2");
}

BOLTZ_TEST(smiles_bracket_charge) {
    auto m = parse_smiles("[O-]");
    expect_eq_i(static_cast<long>(m.atoms.size()), 1, "1 atom");
    expect_eq_i(m.atoms[0].element, 8, "O");
    expect_eq_i(m.atoms[0].charge, -1, "charge -1");

    auto m2 = parse_smiles("[NH4+]");
    expect_eq_i(m2.atoms[0].element, 7, "N");
    expect_eq_i(m2.atoms[0].charge, 1, "charge +1");

    auto m3 = parse_smiles("[Mg+2]");
    expect_eq_i(m3.atoms[0].charge, 2, "charge +2");
}

BOLTZ_TEST(smiles_disconnection) {
    auto m = parse_smiles("C.C");  // two separate carbons, no bond
    expect_eq_i(static_cast<long>(m.atoms.size()), 2, "2 atoms");
    expect_eq_i(static_cast<long>(m.bonds.size()), 0, "no bond across '.'");
}

BOLTZ_TEST(smiles_errors) {
    expect_throws_contains([] { parse_smiles("C("); }, "unbalanced", "open paren");
    expect_throws_contains([] { parse_smiles("C)"); }, "unbalanced", "close paren");
    expect_throws_contains([] { parse_smiles("[O"); }, "unclosed", "open bracket");
    expect_throws_contains([] { parse_smiles("Q"); }, "unknown element", "bad element");
}

int main() {
    std::printf("== test_smiles ==\n");
    return boltztest::run_all();
}
