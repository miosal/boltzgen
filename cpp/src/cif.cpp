#include "boltz/cif.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace boltz {
namespace {

// Tokenize a CIF data line, honouring single/double quoted values.
std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> toks;
    size_t i = 0;
    const size_t n = line.size();
    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (i >= n) break;
        if (line[i] == '\'' || line[i] == '"') {
            const char q = line[i++];
            std::string t;
            while (i < n && line[i] != q) t += line[i++];
            if (i < n) ++i;  // closing quote
            toks.push_back(t);
        } else {
            std::string t;
            while (i < n && !std::isspace(static_cast<unsigned char>(line[i]))) t += line[i++];
            toks.push_back(t);
        }
    }
    return toks;
}

bool starts_with(const std::string& s, const char* p) {
    return s.rfind(p, 0) == 0;
}

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

}  // namespace

std::vector<CifAtom> parse_cif_atoms(const std::string& text) {
    std::istringstream in(text);
    std::string line;
    std::vector<CifAtom> atoms;

    while (std::getline(in, line)) {
        if (trim(line) != "loop_") continue;

        // Collect column headers for this loop.
        std::vector<std::string> headers;
        std::streampos data_start;
        while (std::getline(in, line)) {
            const std::string t = trim(line);
            if (starts_with(t, "_")) {
                headers.push_back(t);
            } else {
                data_start = in.tellg();  // unused; we process `line` below
                break;
            }
        }
        // Only interested in the _atom_site loop.
        bool is_atom_site = false;
        std::unordered_map<std::string, int> col;
        for (size_t k = 0; k < headers.size(); ++k) {
            if (starts_with(headers[k], "_atom_site.")) is_atom_site = true;
            col[headers[k]] = static_cast<int>(k);
        }
        if (!is_atom_site) continue;

        auto idx = [&](const char* name) -> int {
            auto it = col.find(name);
            return it == col.end() ? -1 : it->second;
        };
        const int i_group = idx("_atom_site.group_PDB");
        const int i_atom = idx("_atom_site.label_atom_id");
        const int i_comp = idx("_atom_site.label_comp_id");
        const int i_asym = idx("_atom_site.label_asym_id");
        const int i_seq = idx("_atom_site.label_seq_id");
        const int i_x = idx("_atom_site.Cartn_x");
        const int i_y = idx("_atom_site.Cartn_y");
        const int i_z = idx("_atom_site.Cartn_z");

        // `line` currently holds the first data row.
        do {
            const std::string t = trim(line);
            if (t.empty() || t == "#" || t == "loop_" || starts_with(t, "_")) break;
            std::vector<std::string> f = tokenize(line);
            if (static_cast<int>(f.size()) != static_cast<int>(headers.size())) break;

            CifAtom a;
            if (i_group >= 0) a.group = f[i_group];
            if (i_atom >= 0) a.atom_id = f[i_atom];
            if (i_comp >= 0) a.comp_id = f[i_comp];
            if (i_asym >= 0) a.asym_id = f[i_asym];
            if (i_seq >= 0) a.seq_id = (f[i_seq] == "." || f[i_seq] == "?")
                                          ? -1 : std::atoi(f[i_seq].c_str());
            if (i_x >= 0) a.x = static_cast<float>(std::atof(f[i_x].c_str()));
            if (i_y >= 0) a.y = static_cast<float>(std::atof(f[i_y].c_str()));
            if (i_z >= 0) a.z = static_cast<float>(std::atof(f[i_z].c_str()));
            atoms.push_back(std::move(a));
        } while (std::getline(in, line));
    }
    return atoms;
}

std::vector<CifAtom> parse_cif_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("parse_cif_file: cannot open '" + path + "'");
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_cif_atoms(ss.str());
}

std::vector<Residue> residues_with_ca(const std::vector<CifAtom>& atoms,
                                      std::optional<std::string> asym_id) {
    std::vector<Residue> residues;
    // Key (asym, seq) -> index into residues, preserving first-seen order.
    std::unordered_map<std::string, int> seen;
    for (const CifAtom& a : atoms) {
        if (asym_id.has_value() && a.asym_id != *asym_id) continue;
        const std::string key = a.asym_id + "/" + std::to_string(a.seq_id);
        auto it = seen.find(key);
        int ri;
        if (it == seen.end()) {
            ri = static_cast<int>(residues.size());
            seen[key] = ri;
            Residue r;
            r.asym_id = a.asym_id;
            r.seq_id = a.seq_id;
            r.comp_id = a.comp_id;
            residues.push_back(r);
        } else {
            ri = it->second;
        }
        if (a.atom_id == "CA") {
            residues[ri].ca_x = a.x;
            residues[ri].ca_y = a.y;
            residues[ri].ca_z = a.z;
            residues[ri].has_ca = true;
        }
    }
    return residues;
}

}  // namespace boltz
