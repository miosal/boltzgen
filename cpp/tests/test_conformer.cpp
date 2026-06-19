// Geometric validation of the conformer generator: bonded atoms sit at their
// ideal bond length, non-bonded atoms don't clash, output is deterministic.
// These properties need no RDKit reference data.
#include "boltz/conformer.hpp"
#include "boltz/smiles.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <unordered_set>

using namespace boltz;
using namespace boltztest;

static float dist(const Conformer& c, int a, int b) {
    float d2 = 0;
    for (int k = 0; k < 3; ++k) { float d = c.coords[a * 3 + k] - c.coords[b * 3 + k]; d2 += d * d; }
    return std::sqrt(d2);
}

static void check_geometry(const SmilesMol& m, const Conformer& c, float bond_tol, float clash) {
    std::unordered_set<long long> bonded;
    auto key = [](int a, int b) { return a < b ? (long long)a * 100000 + b : (long long)b * 100000 + a; };
    // Bond lengths within tolerance of ideal.
    for (const auto& b : m.bonds) {
        bonded.insert(key(b.i, b.j));
        const float ideal = ideal_bond_length(m.atoms[b.i].element, m.atoms[b.j].element, b.order, b.aromatic);
        const float d = dist(c, b.i, b.j);
        expect_true(std::fabs(d - ideal) < bond_tol, "bond length near ideal");
    }
    // No non-bonded clashes.
    for (int a = 0; a < c.n; ++a)
        for (int b = a + 1; b < c.n; ++b)
            if (!bonded.count(key(a, b)))
                expect_true(dist(c, a, b) > clash, "no clash");
}

BOLTZ_TEST(ideal_bond_lengths_reasonable) {
    // C-C single ~1.5, C=C ~1.34, C#C ~1.2, aromatic C-C ~1.4 (covalent-radii based).
    expect_true(std::fabs(ideal_bond_length(6, 6, 1, false) - 1.52f) < 0.1f, "C-C");
    expect_true(ideal_bond_length(6, 6, 2, false) < ideal_bond_length(6, 6, 1, false), "C=C shorter");
    expect_true(ideal_bond_length(6, 6, 3, false) < ideal_bond_length(6, 6, 2, false), "C#C shortest");
}

BOLTZ_TEST(conformer_ethane_bond) {
    auto m = parse_smiles("CC");
    auto c = generate_conformer(m, 1, 600);
    expect_eq_i(c.n, 2, "2 atoms");
    const float ideal = ideal_bond_length(6, 6, 1, false);
    expect_true(std::fabs(dist(c, 0, 1) - ideal) < 0.05f, "C-C at ideal");
}

BOLTZ_TEST(conformer_benzene_ring) {
    auto m = parse_smiles("c1ccccc1");
    auto c = generate_conformer(m, 7, 2000);
    expect_eq_i(c.n, 6, "6 atoms");
    check_geometry(m, c, /*bond_tol=*/0.12f, /*clash=*/1.2f);
}

BOLTZ_TEST(conformer_branched) {
    auto m = parse_smiles("CC(C)CO");  // isobutanol-ish heavy atoms
    auto c = generate_conformer(m, 3, 2000);
    check_geometry(m, c, 0.12f, 1.2f);
}

BOLTZ_TEST(conformer_deterministic) {
    auto m = parse_smiles("CCO");
    auto a = generate_conformer(m, 42, 800);
    auto b = generate_conformer(m, 42, 800);
    for (size_t i = 0; i < a.coords.size(); ++i) expect_eq_f(a.coords[i], b.coords[i], "deterministic");
}

BOLTZ_TEST(conformer_centered) {
    auto m = parse_smiles("CCO");
    auto c = generate_conformer(m, 1, 800);
    float cm[3] = {0, 0, 0};
    for (int i = 0; i < c.n; ++i) for (int d = 0; d < 3; ++d) cm[d] += c.coords[i * 3 + d];
    for (int d = 0; d < 3; ++d) expect_true(std::fabs(cm[d]) < 1e-3f, "centered at origin");
}

int main() {
    std::printf("== test_conformer ==\n");
    return boltztest::run_all();
}
