#include "boltz/smiles.hpp"

#include <array>
#include <cctype>
#include <stdexcept>
#include <unordered_map>

namespace boltz {
namespace {

int element_z(const std::string& sym) {
    static const std::unordered_map<std::string, int> z = {
        {"H", 1},  {"B", 5},  {"C", 6},  {"N", 7},  {"O", 8},  {"F", 9},
        {"P", 15}, {"S", 16}, {"Cl", 17}, {"Br", 35}, {"I", 53}, {"Se", 34},
    };
    auto it = z.find(sym);
    return it == z.end() ? 0 : it->second;
}

bool is_aromatic_organic(char c) {
    return c == 'b' || c == 'c' || c == 'n' || c == 'o' || c == 'p' || c == 's';
}

}  // namespace

SmilesMol parse_smiles(const std::string& s) {
    SmilesMol mol;
    std::vector<int> branch_stack;  // saved "prev atom" indices
    // ring_id -> (atom index, pending bond order, aromatic) awaiting closure.
    std::array<int, 100> ring_atom;
    std::array<int, 100> ring_order;
    ring_atom.fill(-1);

    int prev = -1;        // previous atom to bond from (-1 = none)
    int pending = 0;      // pending explicit bond order (0 = default); 4 = aromatic ':'
    size_t i = 0;

    auto add_atom = [&](int element, bool aromatic, int charge) {
        const int idx = static_cast<int>(mol.atoms.size());
        mol.atoms.push_back({element, aromatic, charge});
        if (prev >= 0) {
            SmilesBond b;
            b.i = prev; b.j = idx;
            if (pending == 4) { b.order = 1; b.aromatic = true; }
            else if (pending > 0) { b.order = pending; b.aromatic = false; }
            else if (mol.atoms[prev].aromatic && aromatic) { b.order = 1; b.aromatic = true; }
            else { b.order = 1; b.aromatic = false; }
            mol.bonds.push_back(b);
        }
        pending = 0;
        prev = idx;
    };

    while (i < s.size()) {
        const char c = s[i];
        if (c == '(') { branch_stack.push_back(prev); ++i; }
        else if (c == ')') {
            if (branch_stack.empty()) throw std::invalid_argument("SMILES: unbalanced ')'");
            prev = branch_stack.back(); branch_stack.pop_back(); ++i;
        }
        else if (c == '-') { pending = 1; ++i; }
        else if (c == '=') { pending = 2; ++i; }
        else if (c == '#') { pending = 3; ++i; }
        else if (c == ':') { pending = 4; ++i; }
        else if (c == '/' || c == '\\') { ++i; }  // stereo bond, ignore
        else if (c == '.') { prev = -1; pending = 0; ++i; }  // disconnection
        else if (std::isdigit(static_cast<unsigned char>(c)) || c == '%') {
            int ring = 0;
            if (c == '%') {
                if (i + 2 >= s.size() + 1 || !std::isdigit((unsigned char)s[i + 1]) || !std::isdigit((unsigned char)s[i + 2]))
                    throw std::invalid_argument("SMILES: malformed %nn ring");
                ring = (s[i + 1] - '0') * 10 + (s[i + 2] - '0'); i += 3;
            } else { ring = c - '0'; ++i; }
            if (prev < 0) throw std::invalid_argument("SMILES: ring digit before any atom");
            if (ring_atom[ring] < 0) { ring_atom[ring] = prev; ring_order[ring] = pending; pending = 0; }
            else {
                SmilesBond b; b.i = ring_atom[ring]; b.j = prev;
                const int ord = ring_order[ring] ? ring_order[ring] : pending;
                if (ord == 4) { b.order = 1; b.aromatic = true; }
                else if (ord > 0) { b.order = ord; }
                else if (mol.atoms[b.i].aromatic && mol.atoms[b.j].aromatic) { b.order = 1; b.aromatic = true; }
                mol.bonds.push_back(b);
                ring_atom[ring] = -1; pending = 0;
            }
        }
        else if (c == '[') {
            const size_t end = s.find(']', i);
            if (end == std::string::npos) throw std::invalid_argument("SMILES: unclosed '['");
            std::string inner = s.substr(i + 1, end - i - 1);
            // [isotope]? symbol [H count]? [charge]? : parse symbol + charge.
            size_t p = 0;
            while (p < inner.size() && std::isdigit((unsigned char)inner[p])) ++p;  // isotope
            if (p >= inner.size()) throw std::invalid_argument("SMILES: empty bracket atom");
            bool aromatic = false;
            std::string sym;
            if (std::islower((unsigned char)inner[p])) { aromatic = true; sym = std::string(1, (char)std::toupper((unsigned char)inner[p])); ++p; }
            else {
                sym = std::string(1, inner[p]); ++p;
                if (p < inner.size() && std::islower((unsigned char)inner[p])) { sym += inner[p]; ++p; }
            }
            int charge = 0;
            for (; p < inner.size(); ++p) {
                if (inner[p] == '+' || inner[p] == '-') {
                    const int sign = inner[p] == '+' ? 1 : -1;
                    int mag = 1;
                    if (p + 1 < inner.size() && std::isdigit((unsigned char)inner[p + 1])) { mag = inner[p + 1] - '0'; ++p; }
                    else { while (p + 1 < inner.size() && inner[p + 1] == inner[p]) { ++mag; ++p; } }
                    charge = sign * mag;
                }
                // H counts and other tokens ignored for the heavy-atom graph.
            }
            add_atom(element_z(sym), aromatic, charge);
            i = end + 1;
        }
        else if (std::isalpha(static_cast<unsigned char>(c))) {
            std::string sym(1, c);
            // Two-letter organic elements Cl, Br.
            if ((c == 'C' && i + 1 < s.size() && s[i + 1] == 'l') ||
                (c == 'B' && i + 1 < s.size() && s[i + 1] == 'r')) {
                sym += s[i + 1]; ++i;
            }
            const bool aromatic = is_aromatic_organic(c);
            std::string up = aromatic ? std::string(1, (char)std::toupper((unsigned char)c)) : sym;
            const int z = element_z(up);
            if (z == 0) throw std::invalid_argument("SMILES: unknown element '" + sym + "'");
            add_atom(z, aromatic, 0);
            ++i;
        }
        else throw std::invalid_argument(std::string("SMILES: unexpected char '") + c + "'");
    }
    if (!branch_stack.empty()) throw std::invalid_argument("SMILES: unbalanced '('");
    return mol;
}

}  // namespace boltz
