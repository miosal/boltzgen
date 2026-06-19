// Tests for the hand-rolled mmCIF _atom_site parser: a synthetic CIF with known
// atoms, plus a smoke parse of the real example/inverse_folding/1brs.cif.
#include "boltz/cif.hpp"
#include "test_framework.hpp"

#include <fstream>
#include <string>

using namespace boltz;
using namespace boltztest;

#ifndef BOLTZ_EXAMPLE_DIR
#define BOLTZ_EXAMPLE_DIR "."
#endif

static const char* kSyntheticCif = R"(data_test
loop_
_atom_type.symbol
C
N
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 N VAL A 3 1.0 2.0 3.0
ATOM 2 CA VAL A 3 1.5 2.5 3.5
ATOM 3 C VAL A 3 2.0 3.0 4.0
ATOM 4 N ILE A 4 4.0 4.0 4.0
ATOM 5 CA ILE A 4 4.5 4.5 4.5
ATOM 6 CA GLY D 10 9.0 9.0 9.0
#
)";

BOLTZ_TEST(cif_parses_atoms) {
    auto atoms = parse_cif_atoms(kSyntheticCif);
    expect_eq_i(static_cast<long>(atoms.size()), 6, "atom count");
    expect_true(atoms[1].atom_id == "CA" && atoms[1].comp_id == "VAL", "atom1 CA VAL");
    expect_eq_f(atoms[1].x, 1.5f, "atom1 x");
    expect_eq_f(atoms[1].z, 3.5f, "atom1 z");
    expect_eq_i(atoms[5].seq_id, 10, "atom5 seq");
    expect_true(atoms[5].asym_id == "D", "atom5 chain D");
}

BOLTZ_TEST(cif_residues_with_ca) {
    auto atoms = parse_cif_atoms(kSyntheticCif);
    // Chain A only: VAL3, ILE4.
    auto resA = residues_with_ca(atoms, std::string("A"));
    expect_eq_i(static_cast<long>(resA.size()), 2, "two residues in A");
    expect_true(resA[0].comp_id == "VAL" && resA[0].seq_id == 3, "res0 VAL3");
    expect_true(resA[0].has_ca, "res0 has CA");
    expect_eq_f(resA[0].ca_x, 1.5f, "res0 CA x");
    expect_eq_f(resA[1].ca_y, 4.5f, "res1 CA y");
    // All chains: adds GLY10 on D.
    auto all = residues_with_ca(atoms);
    expect_eq_i(static_cast<long>(all.size()), 3, "three residues total");
}

BOLTZ_TEST(cif_real_example_smoke) {
    const std::string path = std::string(BOLTZ_EXAMPLE_DIR) + "/inverse_folding/1brs.cif";
    std::ifstream f(path);
    if (!f) {
        std::printf("    (skip: %s not found)\n", path.c_str());
        return;
    }
    auto atoms = parse_cif_file(path);
    expect_true(atoms.size() > 1000, "1brs has many atoms");
    // First atom in the file is VAL A 3, N (per the CIF).
    expect_true(atoms[0].comp_id == "VAL", "first residue VAL");
    expect_true(atoms[0].asym_id == "A", "first chain A");
    auto resA = residues_with_ca(atoms, std::string("A"));
    expect_true(resA.size() > 50, "chain A has many residues");
    expect_true(resA[0].has_ca, "chain A res0 has CA");
}

int main() {
    std::printf("== test_cif ==\n");
    return boltztest::run_all();
}
