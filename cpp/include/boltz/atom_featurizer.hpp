// Atom-level featurization for STANDARD residues (the portable part of
// boltzgen.data.feature.featurizer.process_atom_features). The 20 canonical
// amino acids have fixed heavy-atom topology, so atom names / elements /
// atom-to-token mapping / backbone masks can be baked without RDKit. (RDKit is
// only needed for arbitrary-ligand 3-D conformer generation, which is out of
// scope here.) Reference conformer *coordinates* are not baked — those come
// from CCD; for inference on a parsed structure the coordinates come from the
// CIF.
#pragma once

#include <string>
#include <vector>

namespace boltz {

// Heavy-atom (no hydrogen) names for a standard amino acid, in canonical PDB
// order, or empty if `comp_id` is not a standard residue.
const std::vector<std::string>& standard_residue_atoms(const std::string& comp_id);

// Atomic number from a PDB atom name (first alphabetic char: N/C/O/S/P/...).
int element_of_atom(const std::string& atom_name);

struct AtomFeatures {
    int n_atoms = 0;
    std::vector<int> atom_to_token;       // [M] token index of each atom
    std::vector<int> element;             // [M] atomic number
    std::vector<float> backbone_mask;     // [M] 1 if N/CA/C/O
    std::vector<std::string> atom_name;   // [M]
    std::vector<int> token_to_rep_atom;   // [N] representative (CA) atom per token
    std::vector<int> token_atom_start;    // [N] first atom index of token
    std::vector<int> token_atom_count;    // [N] atom count of token
};

// Build atom features for a chain of standard residues (unknown residues are
// treated as a single CA placeholder atom).
AtomFeatures atom_features(const std::vector<std::string>& comp_ids);

}  // namespace boltz
