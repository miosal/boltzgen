#include "boltz/atom_featurizer.hpp"

#include <array>
#include <cctype>
#include <unordered_map>

namespace boltz {
namespace {

// Canonical heavy-atom names for the 20 standard amino acids (PDB nomenclature).
const std::unordered_map<std::string, std::vector<std::string>>& table() {
    static const std::unordered_map<std::string, std::vector<std::string>> t = {
        {"ALA", {"N", "CA", "C", "O", "CB"}},
        {"ARG", {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}},
        {"ASN", {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"}},
        {"ASP", {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"}},
        {"CYS", {"N", "CA", "C", "O", "CB", "SG"}},
        {"GLN", {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"}},
        {"GLU", {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"}},
        {"GLY", {"N", "CA", "C", "O"}},
        {"HIS", {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"}},
        {"ILE", {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"}},
        {"LEU", {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"}},
        {"LYS", {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"}},
        {"MET", {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"}},
        {"PHE", {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}},
        {"PRO", {"N", "CA", "C", "O", "CB", "CG", "CD"}},
        {"SER", {"N", "CA", "C", "O", "CB", "OG"}},
        {"THR", {"N", "CA", "C", "O", "CB", "OG1", "CG2"}},
        {"TRP", {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}},
        {"TYR", {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}},
        {"VAL", {"N", "CA", "C", "O", "CB", "CG1", "CG2"}},
    };
    return t;
}

bool is_backbone(const std::string& a) {
    return a == "N" || a == "CA" || a == "C" || a == "O";
}

}  // namespace

const std::vector<std::string>& standard_residue_atoms(const std::string& comp_id) {
    static const std::vector<std::string> empty;
    auto it = table().find(comp_id);
    return it == table().end() ? empty : it->second;
}

int element_of_atom(const std::string& atom_name) {
    for (char c : atom_name) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            switch (std::toupper(static_cast<unsigned char>(c))) {
                case 'N': return 7;
                case 'C': return 6;
                case 'O': return 8;
                case 'S': return 16;
                case 'P': return 15;
                case 'H': return 1;
                default: return 0;
            }
        }
    }
    return 0;
}

AtomFeatures atom_features(const std::vector<std::string>& comp_ids) {
    const int N = static_cast<int>(comp_ids.size());
    AtomFeatures f;
    f.token_to_rep_atom.assign(N, -1);
    f.token_atom_start.assign(N, 0);
    f.token_atom_count.assign(N, 0);

    for (int t = 0; t < N; ++t) {
        const std::vector<std::string>& atoms = standard_residue_atoms(comp_ids[t]);
        const int start = f.n_atoms;
        f.token_atom_start[t] = start;

        if (atoms.empty()) {
            // Unknown residue: single CA placeholder atom (representative).
            f.atom_name.push_back("CA");
            f.element.push_back(6);
            f.backbone_mask.push_back(1.0f);
            f.atom_to_token.push_back(t);
            f.token_to_rep_atom[t] = f.n_atoms;
            f.n_atoms += 1;
            f.token_atom_count[t] = 1;
            continue;
        }

        for (const std::string& a : atoms) {
            f.atom_name.push_back(a);
            f.element.push_back(element_of_atom(a));
            f.backbone_mask.push_back(is_backbone(a) ? 1.0f : 0.0f);
            f.atom_to_token.push_back(t);
            if (a == "CA") f.token_to_rep_atom[t] = f.n_atoms;
            f.n_atoms += 1;
        }
        if (f.token_to_rep_atom[t] < 0) f.token_to_rep_atom[t] = start;  // fallback
        f.token_atom_count[t] = f.n_atoms - start;
    }
    return f;
}

}  // namespace boltz
