// Hand-rolled mmCIF writer (output step). Serializes atoms to a minimal valid
// mmCIF `_atom_site` loop — the inverse of cif.hpp's parser, with no gemmi.
// Used to write designed/predicted structures back out.
#pragma once

#include "boltz/cif.hpp"

#include <string>
#include <vector>

namespace boltz {

// Serialize atoms to an mmCIF string (data block + _atom_site loop).
std::string write_cif(const std::vector<CifAtom>& atoms,
                      const std::string& block_name = "boltzcpp");

// Write to a file.
void write_cif_file(const std::string& path, const std::vector<CifAtom>& atoms,
                    const std::string& block_name = "boltzcpp");

}  // namespace boltz
