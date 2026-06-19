// Minimal SMILES parser -> heavy-atom topology, the portable front-end of the
// ligand path (boltzgen uses rdkit.Chem.MolFromSmiles then RemoveHs). Produces
// the atom/bond graph (elements, aromatic flags, formal charges, bond orders)
// that atom featurization needs. Hydrogens are not materialised (the pipeline
// removes them anyway). 3-D conformer generation (ETKDGv3+UFF) is a separate,
// data/algorithm-heavy step and is NOT done here.
//
// Supports: organic-subset atoms (B,C,N,O,P,S,F,Cl,Br,I), aromatic lowercase
// (b,c,n,o,p,s), bracket atoms [..] with charge, bonds (-=#:), branches (),
// ring-closure digits and %nn, and '.' disconnections. Stereo/isotope markers
// are tolerated and ignored.
#pragma once

#include <string>
#include <vector>

namespace boltz {

struct SmilesAtom {
    int element = 0;     // atomic number
    bool aromatic = false;
    int charge = 0;
};

struct SmilesBond {
    int i = 0;
    int j = 0;
    int order = 1;        // 1 single, 2 double, 3 triple
    bool aromatic = false;
};

struct SmilesMol {
    std::vector<SmilesAtom> atoms;
    std::vector<SmilesBond> bonds;
};

// Parses `smiles`; throws std::invalid_argument on malformed input.
SmilesMol parse_smiles(const std::string& smiles);

}  // namespace boltz
