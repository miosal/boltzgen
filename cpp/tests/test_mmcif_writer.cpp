// Round-trip test for the mmCIF writer: write atoms, re-parse with the CIF
// parser, and confirm the fields survive. Validates the output step end to end.
#include "boltz/cif.hpp"
#include "boltz/mmcif_writer.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

static CifAtom mk(const char* g, const char* atom, const char* comp, const char* chain,
                  int seq, float x, float y, float z) {
    CifAtom a;
    a.group = g; a.atom_id = atom; a.comp_id = comp; a.asym_id = chain;
    a.seq_id = seq; a.x = x; a.y = y; a.z = z;
    return a;
}

BOLTZ_TEST(mmcif_writer_roundtrips) {
    std::vector<CifAtom> atoms = {
        mk("ATOM", "N", "VAL", "A", 3, 1.234f, -5.678f, 9.012f),
        mk("ATOM", "CA", "VAL", "A", 3, 2.0f, 2.0f, 2.0f),
        mk("ATOM", "CA", "GLY", "B", 7, -3.5f, 0.25f, 100.125f),
        mk("HETATM", "ZN", "ZN", "C", -1, 0.0f, 0.0f, 0.0f),
    };

    std::string text = write_cif(atoms, "test");
    auto parsed = parse_cif_atoms(text);

    expect_eq_i(static_cast<long>(parsed.size()), 4, "atom count round-trips");
    for (size_t i = 0; i < atoms.size(); ++i) {
        expect_true(parsed[i].group == atoms[i].group, "group");
        expect_true(parsed[i].atom_id == atoms[i].atom_id, "atom_id");
        expect_true(parsed[i].comp_id == atoms[i].comp_id, "comp_id");
        expect_true(parsed[i].asym_id == atoms[i].asym_id, "asym_id");
        expect_eq_i(parsed[i].seq_id, atoms[i].seq_id, "seq_id");
        // coordinates written at %.3f precision
        expect_true(std::fabs(parsed[i].x - atoms[i].x) < 1e-3f, "x");
        expect_true(std::fabs(parsed[i].y - atoms[i].y) < 1e-3f, "y");
        expect_true(std::fabs(parsed[i].z - atoms[i].z) < 1e-3f, "z");
    }
}

BOLTZ_TEST(mmcif_writer_residues_roundtrip) {
    std::vector<CifAtom> atoms = {
        mk("ATOM", "CA", "ALA", "A", 1, 0, 0, 0),
        mk("ATOM", "CA", "LEU", "A", 2, 3.8f, 0, 0),
    };
    auto parsed = parse_cif_atoms(write_cif(atoms));
    auto res = residues_with_ca(parsed, std::string("A"));
    expect_eq_i(static_cast<long>(res.size()), 2, "two residues");
    expect_true(res[1].comp_id == "LEU" && res[1].has_ca, "LEU CA");
    expect_true(std::fabs(res[1].ca_x - 3.8f) < 1e-3f, "CA x");
}

int main() {
    std::printf("== test_mmcif_writer ==\n");
    return boltztest::run_all();
}
