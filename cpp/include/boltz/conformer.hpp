// 3-D conformer generation for a molecular graph (the step boltzgen does via
// RDKit ETKDGv3 + UFF). This is a hand-rolled distance-geometry / position-based
// embedding: it places atoms so that bonded pairs sit at ideal bond lengths and
// no atoms clash, yielding a chemically-plausible conformer used as reference /
// initial coordinates by atom featurization.
//
// NOTE: this is NOT bit-identical to RDKit's ETKDGv3+UFF (which uses a torsion
// knowledge base and the full UFF force field). It is validated on geometric
// grounds — bond lengths within tolerance, no clashes, connectivity preserved —
// which needs no external reference data. Deterministic given a seed.
#pragma once

#include "boltz/smiles.hpp"

#include <vector>

namespace boltz {

// Ideal bond length (Angstrom) from covalent radii + an order/aromatic factor.
float ideal_bond_length(int element_i, int element_j, int order, bool aromatic);

struct Conformer {
    int n = 0;
    std::vector<float> coords;  // [n*3]
};

// Generate a conformer for `mol`. `iters` relaxation steps; `seed` controls the
// random initial placement.
Conformer generate_conformer(const SmilesMol& mol, unsigned seed = 1, int iters = 1200);

}  // namespace boltz
