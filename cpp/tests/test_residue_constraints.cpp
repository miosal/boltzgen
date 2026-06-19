// C++ port of tests/test_residue_constraints.py — exercises parse_range,
// normalize_aa_spec, convert_aa_names_to_indices and parse_residue_constraints.
#include "boltz/const.hpp"
#include "boltz/residue_constraints.hpp"
#include "test_framework.hpp"

#include <vector>

using namespace boltz;
using namespace boltztest;

static int CI(const char* code) { return canonical_index(code); }

// Convenience builders mirroring the YAML dict specs.
static ConstraintSpec allowed_at(const std::string& pos, AaSpec aa) {
    ConstraintSpec s; s.has_position = true; s.position = pos; s.allowed = std::move(aa); return s;
}
static ConstraintSpec disallowed_at(const std::string& pos, AaSpec aa) {
    ConstraintSpec s; s.has_position = true; s.position = pos; s.disallowed = std::move(aa); return s;
}

// ---- _normalize_aa_spec ----------------------------------------------------
BOLTZ_TEST(normalize_single_letter) {
    auto r = normalize_aa_spec(AaSpec::of_string("A"));
    expect_eq_i(r.size(), 1, "len");
    expect_true(r[0] == "A", "A");
}
BOLTZ_TEST(normalize_multi_letter_string) {
    auto r = normalize_aa_spec(AaSpec::of_string("AGS"));
    expect_eq_i(r.size(), 3, "len");
    expect_true(r[0] == "A" && r[1] == "G" && r[2] == "S", "AGS split");
}
BOLTZ_TEST(normalize_three_letter_code) {
    auto r = normalize_aa_spec(AaSpec::of_string("ALA"));
    expect_eq_i(r.size(), 1, "len");
    expect_true(r[0] == "ALA", "ALA kept");
}
BOLTZ_TEST(normalize_three_letter_not_valid) {
    auto r = normalize_aa_spec(AaSpec::of_string("AGS"));  // 3 chars, not a code
    expect_eq_i(r.size(), 3, "split");
}
BOLTZ_TEST(normalize_list_three_letter) {
    auto r = normalize_aa_spec(AaSpec::of_list({"ALA", "GLY"}));
    expect_true(r.size() == 2 && r[0] == "ALA" && r[1] == "GLY", "list kept");
}
BOLTZ_TEST(normalize_lowercase) {
    auto r = normalize_aa_spec(AaSpec::of_string("ags"));
    expect_true(r.size() == 3 && r[0] == "A", "lower upcased+split");
}
BOLTZ_TEST(normalize_whitespace_stripped) {
    auto r = normalize_aa_spec(AaSpec::of_string(" AG "));
    expect_true(r.size() == 2 && r[0] == "A" && r[1] == "G", "stripped");
}
BOLTZ_TEST(normalize_invalid_type_raises) {
    expect_throws_contains([] { normalize_aa_spec(AaSpec::invalid()); },
                           "Invalid amino acid specification", "invalid type");
}

// ---- _convert_aa_names_to_indices -----------------------------------------
BOLTZ_TEST(convert_single_letter_a) {
    auto r = convert_aa_names_to_indices({"A"});
    expect_eq_i(r[0], CI("ALA"), "A->ALA");
}
BOLTZ_TEST(convert_three_letter) {
    auto r = convert_aa_names_to_indices({"ALA", "GLY"});
    expect_true(r[0] == CI("ALA") && r[1] == CI("GLY"), "codes");
}
BOLTZ_TEST(convert_mixed) {
    auto r = convert_aa_names_to_indices({"A", "GLY"});
    expect_true(r[0] == CI("ALA") && r[1] == CI("GLY"), "mixed");
}
BOLTZ_TEST(convert_all_20_unique) {
    std::vector<std::string> letters;
    for (char c : std::string("ACDEFGHIKLMNPQRSTVWY")) letters.emplace_back(1, c);
    auto r = convert_aa_names_to_indices(letters);
    expect_eq_i(r.size(), 20, "count");
    std::vector<int> seen(20, 0);
    for (int i : r) seen[i] = 1;
    int uniq = 0;
    for (int s : seen) uniq += s;
    expect_eq_i(uniq, 20, "unique");
}
BOLTZ_TEST(convert_invalid_letter_raises) {
    expect_throws_contains([] { convert_aa_names_to_indices({"X"}); },
                           "Unknown amino acid", "X invalid");
}
BOLTZ_TEST(convert_invalid_three_letter_raises) {
    expect_throws_contains([] { convert_aa_names_to_indices({"ZZZ"}); },
                           "Unknown amino acid", "ZZZ invalid");
}

// ---- parse_residue_constraints: valid -------------------------------------
BOLTZ_TEST(prc_empty_list_zeros) {
    auto m = parse_residue_constraints({}, 10);
    expect_true(m.rows == 10 && m.cols == 20, "shape");
    float s = 0;
    for (float v : m.data) s += v;
    expect_eq_f(s, 0.0f, "sum zero");
}
BOLTZ_TEST(prc_single_allowed) {
    auto m = parse_residue_constraints({allowed_at("1", AaSpec::of_string("A"))}, 5);
    expect_eq_f(m.at(0, CI("ALA")), 0.0f, "ALA allowed");
    expect_eq_f(m.row_sum(0), 19.0f, "19 blocked");
    for (int r = 1; r < 5; ++r) expect_eq_f(m.row_sum(r), 0.0f, "rest untouched");
}
BOLTZ_TEST(prc_single_disallowed) {
    auto m = parse_residue_constraints({disallowed_at("3", AaSpec::of_string("CM"))}, 5);
    expect_eq_f(m.at(2, CI("CYS")), 1.0f, "CYS blocked");
    expect_eq_f(m.at(2, CI("MET")), 1.0f, "MET blocked");
    expect_eq_f(m.row_sum(2), 2.0f, "two blocked");
}
BOLTZ_TEST(prc_range_positions) {
    auto m = parse_residue_constraints({disallowed_at("3..5", AaSpec::of_string("C"))}, 10);
    for (int p : {2, 3, 4}) expect_eq_f(m.at(p, CI("CYS")), 1.0f, "range blocked");
    for (int p : {0, 1, 5, 6, 7, 8, 9}) expect_eq_f(m.at(p, CI("CYS")), 0.0f, "outside");
}
BOLTZ_TEST(prc_allowed_multiple) {
    auto m = parse_residue_constraints({allowed_at("8", AaSpec::of_string("AGS"))}, 10);
    expect_eq_f(m.at(7, CI("ALA")), 0.0f, "A");
    expect_eq_f(m.at(7, CI("GLY")), 0.0f, "G");
    expect_eq_f(m.at(7, CI("SER")), 0.0f, "S");
    expect_eq_f(m.row_sum(7), 17.0f, "17 blocked");
}
BOLTZ_TEST(prc_list_format_allowed) {
    auto m = parse_residue_constraints({allowed_at("1", AaSpec::of_list({"A", "G"}))}, 5);
    expect_eq_f(m.at(0, CI("ALA")), 0.0f, "A");
    expect_eq_f(m.at(0, CI("GLY")), 0.0f, "G");
    expect_eq_f(m.row_sum(0), 18.0f, "18 blocked");
}
BOLTZ_TEST(prc_overlapping_allowed_intersection) {
    auto m = parse_residue_constraints(
        {allowed_at("1", AaSpec::of_string("AG")), allowed_at("1", AaSpec::of_string("GS"))}, 5);
    expect_eq_f(m.at(0, CI("GLY")), 0.0f, "G survives");
    expect_eq_f(m.at(0, CI("ALA")), 1.0f, "A blocked");
    expect_eq_f(m.at(0, CI("SER")), 1.0f, "S blocked");
    expect_eq_f(m.row_sum(0), 19.0f, "only G");
}
BOLTZ_TEST(prc_overlapping_range_intersection) {
    auto m = parse_residue_constraints(
        {allowed_at("1..5", AaSpec::of_string("AG")), allowed_at("3..7", AaSpec::of_string("GS"))}, 10);
    expect_eq_f(m.row_sum(0), 18.0f, "pos1 AG");
    for (int p : {2, 3, 4}) {
        expect_eq_f(m.at(p, CI("GLY")), 0.0f, "G");
        expect_eq_f(m.at(p, CI("ALA")), 1.0f, "A blocked");
        expect_eq_f(m.row_sum(p), 19.0f, "only G");
    }
    expect_eq_f(m.row_sum(5), 18.0f, "pos6 GS");
}
BOLTZ_TEST(prc_allowed_then_disallowed) {
    auto m = parse_residue_constraints(
        {allowed_at("5", AaSpec::of_string("AGILMV")), disallowed_at("5", AaSpec::of_string("CM"))}, 10);
    expect_eq_f(m.at(4, CI("MET")), 1.0f, "M blocked");
    expect_eq_f(m.at(4, CI("ALA")), 0.0f, "A allowed");
}
BOLTZ_TEST(prc_order_independent) {
    auto ab = parse_residue_constraints(
        {allowed_at("5", AaSpec::of_string("AGILMV")), disallowed_at("5", AaSpec::of_string("CM"))}, 10);
    auto ba = parse_residue_constraints(
        {disallowed_at("5", AaSpec::of_string("CM")), allowed_at("5", AaSpec::of_string("AGILMV"))}, 10);
    expect_true(ab.data == ba.data, "order independent");
}
BOLTZ_TEST(prc_disjoint_allowed_all_blocked) {
    auto m = parse_residue_constraints(
        {allowed_at("1", AaSpec::of_string("AG")), allowed_at("1", AaSpec::of_string("VILM"))}, 5);
    expect_eq_f(m.row_sum(0), 20.0f, "all blocked");
}
BOLTZ_TEST(prc_multiple_disallowed_accumulate) {
    auto m = parse_residue_constraints(
        {disallowed_at("1", AaSpec::of_string("CM")), disallowed_at("1", AaSpec::of_string("WK"))}, 5);
    for (const char* c : {"CYS", "MET", "TRP", "LYS"}) expect_eq_f(m.at(0, CI(c)), 1.0f, "blocked");
    expect_eq_f(m.row_sum(0), 4.0f, "four blocked");
}
BOLTZ_TEST(prc_dtype_and_shape) {
    auto m = parse_residue_constraints({allowed_at("1", AaSpec::of_string("A"))}, 10);
    expect_true(m.rows == 10 && m.cols == 20, "shape");
}

// ---- parse_residue_constraints: errors ------------------------------------
BOLTZ_TEST(prc_missing_position) {
    ConstraintSpec s; s.allowed = AaSpec::of_string("A");  // has_position=false
    expect_throws_contains([&] { parse_residue_constraints({s}, 10); },
                           "required", "missing position");
}
BOLTZ_TEST(prc_oob_high) {
    expect_throws_contains([] { parse_residue_constraints({allowed_at("11", AaSpec::of_string("A"))}, 10); },
                           "out of bounds", "oob high");
}
BOLTZ_TEST(prc_oob_zero) {
    expect_throws_contains([] { parse_residue_constraints({allowed_at("0", AaSpec::of_string("A"))}, 10); },
                           "1 indexed", "position zero");
}
BOLTZ_TEST(prc_both_allowed_disallowed) {
    ConstraintSpec s; s.has_position = true; s.position = "1";
    s.allowed = AaSpec::of_string("A"); s.disallowed = AaSpec::of_string("C");
    expect_throws_contains([&] { parse_residue_constraints({s}, 10); },
                           "cannot specify both", "both");
}
BOLTZ_TEST(prc_neither) {
    ConstraintSpec s; s.has_position = true; s.position = "1";
    expect_throws_contains([&] { parse_residue_constraints({s}, 10); },
                           "must specify either", "neither");
}
BOLTZ_TEST(prc_empty_allowed) {
    expect_throws_contains([] { parse_residue_constraints({allowed_at("1", AaSpec::of_string(""))}, 10); },
                           "cannot be empty", "empty allowed");
}
BOLTZ_TEST(prc_invalid_aa) {
    expect_throws_contains([] { parse_residue_constraints({allowed_at("1", AaSpec::of_string("X"))}, 10); },
                           "Unknown amino acid", "invalid aa");
}
BOLTZ_TEST(prc_invalid_aa_disallowed) {
    expect_throws_contains([] { parse_residue_constraints({disallowed_at("1", AaSpec::of_string("XZ"))}, 10); },
                           "Unknown amino acid", "invalid aa disallowed");
}

// ---- regression: original YAML --------------------------------------------
BOLTZ_TEST(prc_original_yaml) {
    auto m = parse_residue_constraints(
        {allowed_at("1", AaSpec::of_string("A")),
         disallowed_at("3..5", AaSpec::of_string("CM")),
         allowed_at("8", AaSpec::of_string("AGS")),
         allowed_at("10", AaSpec::of_string("P"))},
        10);
    expect_eq_f(m.at(0, CI("ALA")), 0.0f, "pos1 A");
    expect_eq_f(m.row_sum(0), 19.0f, "pos1 19");
    for (int p : {2, 3, 4}) {
        expect_eq_f(m.at(p, CI("CYS")), 1.0f, "CM");
        expect_eq_f(m.at(p, CI("MET")), 1.0f, "CM");
        expect_eq_f(m.row_sum(p), 2.0f, "two");
    }
    expect_eq_f(m.row_sum(7), 17.0f, "pos8");
    expect_eq_f(m.at(9, CI("PRO")), 0.0f, "pos10 P");
    expect_eq_f(m.row_sum(9), 19.0f, "pos10");
    for (int p : {1, 5, 6, 8}) expect_eq_f(m.row_sum(p), 0.0f, "unconstrained");
}

int main() {
    std::printf("== test_residue_constraints ==\n");
    return boltztest::run_all();
}
