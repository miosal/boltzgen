// Hand-rolled port of constants from boltzgen.data.const that the inference
// front-end needs. Kept minimal and in sync with src/boltzgen/data/const.py.
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace boltz {

// 20 canonical amino acids, order matches const.canonical_tokens.
inline const std::vector<std::string>& canonical_tokens() {
    static const std::vector<std::string> v = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"};
    return v;
}

// 1-letter -> 3-letter, matches const.prot_letter_to_token (incl. ambiguous
// codes mapping to UNK, and "-" -> "-").
inline const std::unordered_map<std::string, std::string>& prot_letter_to_token() {
    static const std::unordered_map<std::string, std::string> m = {
        {"A", "ALA"}, {"R", "ARG"}, {"N", "ASN"}, {"D", "ASP"}, {"C", "CYS"},
        {"E", "GLU"}, {"Q", "GLN"}, {"G", "GLY"}, {"H", "HIS"}, {"I", "ILE"},
        {"L", "LEU"}, {"K", "LYS"}, {"M", "MET"}, {"F", "PHE"}, {"P", "PRO"},
        {"S", "SER"}, {"T", "THR"}, {"W", "TRP"}, {"Y", "TYR"}, {"V", "VAL"},
        {"X", "UNK"}, {"J", "UNK"}, {"B", "UNK"}, {"Z", "UNK"}, {"O", "UNK"},
        {"U", "UNK"}, {"-", "-"}};
    return m;
}

// Index of a canonical 3-letter code, or -1 if absent (mirrors list.index but
// non-throwing; callers decide how to react).
inline int canonical_index(const std::string& code) {
    const auto& v = canonical_tokens();
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        if (v[i] == code) return i;
    }
    return -1;
}

constexpr int kCanonicalsOffset = 2;  // const.canonicals_offset

}  // namespace boltz
