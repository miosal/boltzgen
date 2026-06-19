#include "boltz/conformer.hpp"

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace boltz {
namespace {

float covalent_radius(int z) {
    static const std::unordered_map<int, float> r = {
        {1, 0.31f},  {5, 0.84f},  {6, 0.76f},  {7, 0.71f},  {8, 0.66f},
        {9, 0.57f},  {15, 1.07f}, {16, 1.05f}, {17, 1.02f}, {34, 1.20f},
        {35, 1.20f}, {53, 1.39f},
    };
    auto it = r.find(z);
    return it == r.end() ? 0.77f : it->second;
}

uint64_t lcg(uint64_t& s) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s; }
float urand(uint64_t& s) { return static_cast<float>((lcg(s) >> 40) & 0xFFFFFF) / 16777216.0f; }

}  // namespace

float ideal_bond_length(int zi, int zj, int order, bool aromatic) {
    const float base = covalent_radius(zi) + covalent_radius(zj);
    float factor = 1.0f;
    if (aromatic) factor = 0.93f;
    else if (order == 2) factor = 0.90f;
    else if (order == 3) factor = 0.83f;
    return base * factor;
}

Conformer generate_conformer(const SmilesMol& mol, unsigned seed, int iters) {
    const int n = static_cast<int>(mol.atoms.size());
    Conformer c;
    c.n = n;
    c.coords.assign((size_t)n * 3, 0.0f);
    if (n == 0) return c;
    if (n == 1) return c;  // single atom at origin

    // Per-bond ideal lengths and a bonded-pair set (for clash exclusion).
    std::vector<float> blen(mol.bonds.size());
    std::unordered_set<long long> bonded;
    auto key = [](int a, int b) { return a < b ? (long long)a * 100000 + b : (long long)b * 100000 + a; };
    for (size_t k = 0; k < mol.bonds.size(); ++k) {
        const SmilesBond& b = mol.bonds[k];
        blen[k] = ideal_bond_length(mol.atoms[b.i].element, mol.atoms[b.j].element, b.order, b.aromatic);
        bonded.insert(key(b.i, b.j));
    }

    // Random initial placement in a box scaled by molecule size.
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
    const float box = 1.5f * std::cbrt((float)n) + 1.0f;
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) c.coords[i * 3 + d] = (urand(s) - 0.5f) * 2.0f * box;

    const float clash = 1.5f;  // min non-bonded distance (A)
    auto* p = c.coords.data();
    auto dist = [&](int a, int b, float* dv) {
        float d2 = 0;
        for (int k = 0; k < 3; ++k) { dv[k] = p[a * 3 + k] - p[b * 3 + k]; d2 += dv[k] * dv[k]; }
        return std::sqrt(d2 + 1e-12f);
    };

    for (int it = 0; it < iters; ++it) {
        // Bond length constraints (position-based: split correction between endpoints).
        for (size_t k = 0; k < mol.bonds.size(); ++k) {
            const int a = mol.bonds[k].i, b = mol.bonds[k].j;
            float dv[3]; const float d = dist(a, b, dv);
            const float corr = (d - blen[k]) / d * 0.5f;
            for (int t = 0; t < 3; ++t) { p[a * 3 + t] -= corr * dv[t]; p[b * 3 + t] += corr * dv[t]; }
        }
        // Soft repulsion to remove clashes between non-bonded atoms.
        for (int a = 0; a < n; ++a)
            for (int b = a + 1; b < n; ++b) {
                if (bonded.count(key(a, b))) continue;
                float dv[3]; const float d = dist(a, b, dv);
                if (d < clash) {
                    const float corr = (d - clash) / d * 0.5f * 0.6f;  // gentle
                    for (int t = 0; t < 3; ++t) { p[a * 3 + t] -= corr * dv[t]; p[b * 3 + t] += corr * dv[t]; }
                }
            }
    }

    // Center at origin.
    float cm[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) cm[d] += p[i * 3 + d];
    for (int d = 0; d < 3; ++d) cm[d] /= n;
    for (int i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) p[i * 3 + d] -= cm[d];
    return c;
}

}  // namespace boltz
