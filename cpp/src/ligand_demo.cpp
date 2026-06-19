// boltzcpp_ligand: small-molecule featurization pipeline — SMILES in, 3-D
// structure out. Demonstrates the arbitrary-chemistry input path end to end:
//
//   SMILES -> parse_smiles (atom/bond graph) -> generate_conformer (3-D coords)
//          -> atom features (element, charge, bonds) -> mmCIF
//
// This is the ligand front-end that feeds the design/folding models (the
// conformer provides reference/initial atom coordinates). Runs on any SMILES.
#include "boltz/cif.hpp"
#include "boltz/conformer.hpp"
#include "boltz/mmcif_writer.hpp"
#include "boltz/smiles.hpp"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace boltz;

static std::string element_symbol(int z) {
    switch (z) {
        case 1: return "H"; case 5: return "B"; case 6: return "C"; case 7: return "N";
        case 8: return "O"; case 9: return "F"; case 15: return "P"; case 16: return "S";
        case 17: return "Cl"; case 34: return "Se"; case 35: return "Br"; case 53: return "I";
        default: return "X";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <SMILES> [--out path]\n", argv[0]);
        return 2;
    }
    const std::string smiles = argv[1];
    std::string out_path;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--out" && i + 1 < argc) out_path = argv[++i];

    SmilesMol mol;
    try {
        mol = parse_smiles(smiles);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SMILES parse error: %s\n", e.what());
        return 1;
    }
    const Conformer conf = generate_conformer(mol, /*seed=*/1, /*iters=*/2000);

    // Bond-order histogram + aromatic count.
    int n_arom = 0, n_double = 0, n_triple = 0;
    for (const auto& b : mol.bonds) {
        if (b.aromatic) ++n_arom;
        else if (b.order == 2) ++n_double;
        else if (b.order == 3) ++n_triple;
    }

    std::printf("SMILES    : %s\n", smiles.c_str());
    std::printf("atoms     : %d  bonds : %d  (aromatic %d, double %d, triple %d)\n",
                (int)mol.atoms.size(), (int)mol.bonds.size(), n_arom, n_double, n_triple);
    std::printf("conformer : 3-D coords generated (bond-length geometry)\n");

    if (!out_path.empty()) {
        std::vector<CifAtom> atoms;
        std::map<int, int> per_elem;  // element -> running count for naming
        for (int i = 0; i < (int)mol.atoms.size(); ++i) {
            const int z = mol.atoms[i].element;
            const std::string sym = element_symbol(z);
            const int idx = ++per_elem[z];
            CifAtom a;
            a.group = "HETATM";
            a.atom_id = sym + std::to_string(idx);
            a.comp_id = "LIG";
            a.asym_id = "X";
            a.seq_id = 1;
            a.x = conf.coords[i * 3]; a.y = conf.coords[i * 3 + 1]; a.z = conf.coords[i * 3 + 2];
            atoms.push_back(a);
        }
        write_cif_file(out_path, atoms, "boltzcpp_ligand");
        std::printf("output    : %s\n", out_path.c_str());
    }
    return 0;
}
