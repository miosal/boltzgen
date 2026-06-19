// Hand-rolled minimal mmCIF parser for the _atom_site loop — enough to drive
// inference (chains, residues, atom names, coordinates). No gemmi. Mirrors the
// subset of boltzgen.data.parse.mmcif that inference actually consumes:
// label_asym_id (chain), label_seq_id (residue index), label_comp_id (residue
// name), label_atom_id (atom name) and Cartn_x/y/z.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace boltz {

struct CifAtom {
    std::string group;     // ATOM / HETATM
    std::string atom_id;   // label_atom_id, e.g. "CA"
    std::string comp_id;   // label_comp_id, e.g. "VAL"
    std::string asym_id;   // label_asym_id, chain
    int seq_id = -1;       // label_seq_id (-1 if '.'/'?')
    float x = 0, y = 0, z = 0;
};

// Parse all _atom_site rows from mmCIF text.
std::vector<CifAtom> parse_cif_atoms(const std::string& text);

// Convenience: read a file and parse it.
std::vector<CifAtom> parse_cif_file(const std::string& path);

// One residue with its representative (CA, or first atom) coordinate.
struct Residue {
    std::string asym_id;
    int seq_id = -1;
    std::string comp_id;
    float ca_x = 0, ca_y = 0, ca_z = 0;
    bool has_ca = false;
};

// Collapse atoms of a chain into ordered residues (by first appearance), taking
// the CA coordinate as the representative center. If `asym_id` is set, restrict
// to that chain; otherwise include all chains.
std::vector<Residue> residues_with_ca(const std::vector<CifAtom>& atoms,
                                      std::optional<std::string> asym_id = std::nullopt);

}  // namespace boltz
