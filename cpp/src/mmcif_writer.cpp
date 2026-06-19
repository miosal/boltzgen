#include "boltz/mmcif_writer.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace boltz {
namespace {

// Element guess: first alphabetic character of the atom name (sufficient for a
// valid type_symbol; not parsed back).
std::string element_of(const std::string& atom_id) {
    for (char c : atom_id)
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
            return std::string(1, static_cast<char>(std::toupper(c)));
    return "X";
}

std::string fmt_coord(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", v);
    return buf;
}

}  // namespace

std::string write_cif(const std::vector<CifAtom>& atoms, const std::string& block_name) {
    std::ostringstream out;
    out << "data_" << block_name << "\n#\n";
    out << "loop_\n";
    out << "_atom_site.group_PDB\n";
    out << "_atom_site.id\n";
    out << "_atom_site.type_symbol\n";
    out << "_atom_site.label_atom_id\n";
    out << "_atom_site.label_comp_id\n";
    out << "_atom_site.label_asym_id\n";
    out << "_atom_site.label_seq_id\n";
    out << "_atom_site.Cartn_x\n";
    out << "_atom_site.Cartn_y\n";
    out << "_atom_site.Cartn_z\n";
    out << "_atom_site.occupancy\n";
    out << "_atom_site.B_iso_or_equiv\n";
    out << "_atom_site.pdbx_PDB_model_num\n";

    int id = 1;
    for (const CifAtom& a : atoms) {
        const std::string group = a.group.empty() ? "ATOM" : a.group;
        const std::string seq = a.seq_id < 0 ? "." : std::to_string(a.seq_id);
        out << group << ' ' << id++ << ' ' << element_of(a.atom_id) << ' '
            << a.atom_id << ' ' << a.comp_id << ' ' << a.asym_id << ' ' << seq << ' '
            << fmt_coord(a.x) << ' ' << fmt_coord(a.y) << ' ' << fmt_coord(a.z)
            << " 1.00 0.00 1\n";
    }
    out << "#\n";
    return out.str();
}

void write_cif_file(const std::string& path, const std::vector<CifAtom>& atoms,
                    const std::string& block_name) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("write_cif_file: cannot open '" + path + "'");
    f << write_cif(atoms, block_name);
}

}  // namespace boltz
