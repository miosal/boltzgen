// C++ port of tests/test_inverse_fold_constraint_masks.py — exercises
// build_constraint_logit_mask composition of per-residue + global restrictions.
#include "boltz/const.hpp"
#include "boltz/residue_constraints.hpp"
#include "test_framework.hpp"

using namespace boltz;
using namespace boltztest;

static constexpr float INF = 1e6f;

static int CI(const char* code) { return canonical_index(code); }

// Single-row (1 x 20) mask where only `allowed` tokens are permitted (0.0),
// everything else blocked (1.0) — mirrors _allowed_only_mask in the Python test.
static AaConstraintMaskInput allowed_only_mask(const std::vector<std::string>& allowed) {
    AaConstraintMaskInput in;
    in.present = true;
    in.rows = 1;
    in.cols = 20;
    in.data.assign(20, 1.0f);
    for (const std::string& t : allowed) in.data[canonical_index(t)] = 0.0f;
    return in;
}

static bool has_warning(const LogitMaskResult& r, const std::string& needle) {
    for (const std::string& w : r.warnings)
        if (w.find(needle) != std::string::npos) return true;
    return false;
}

static int count_zero(const LogitMaskResult& r, int row) {
    int n = 0;
    for (int c = 0; c < r.cols; ++c)
        if (r.at(row, c) == 0.0f) ++n;
    return n;
}

BOLTZ_TEST(conflict_allowed_and_global_avoid_keeps_global) {
    auto mask = allowed_only_mask({"CYS"});
    auto out = build_constraint_logit_mask(1, mask, {"CYS"}, INF);
    expect_true(has_warning(out, "Relaxing per-residue constraints"), "warns relax");
    expect_eq_f(out.at(0, CI("CYS")), -INF, "CYS still blocked");
    expect_eq_i(count_zero(out, 0), 19, "others available");
}

BOLTZ_TEST(non_conflicting_constraints_compose) {
    auto mask = allowed_only_mask({"ALA"});
    auto out = build_constraint_logit_mask(1, mask, {"CYS"}, INF);
    expect_eq_f(out.at(0, CI("ALA")), 0.0f, "ALA allowed");
    expect_eq_f(out.at(0, CI("CYS")), -INF, "CYS blocked");
    expect_eq_i(count_zero(out, 0), 1, "only ALA");
}

BOLTZ_TEST(global_restrictions_block_all_raise) {
    AaConstraintMaskInput none;  // present=false  (aa_constraint_mask=None)
    expect_throws_contains(
        [&] { build_constraint_logit_mask(1, none, canonical_tokens(), INF); },
        "no valid amino acids", "all blocked raises");
}

BOLTZ_TEST(shape_mismatch_ignores_per_residue) {
    AaConstraintMaskInput bad;
    bad.present = true;
    bad.rows = 2;  // mismatch vs num_nodes=1
    bad.cols = 20;
    bad.data.assign(40, 0.0f);
    auto out = build_constraint_logit_mask(1, bad, {}, INF);
    expect_true(has_warning(out, "shape mismatch"), "warns mismatch");
    expect_true(out.rows == 1 && out.cols == 20, "shape");
    for (float v : out.data) expect_eq_f(v, 0.0f, "all zero");
}

int main() {
    std::printf("== test_inverse_fold_constraint_masks ==\n");
    return boltztest::run_all();
}
