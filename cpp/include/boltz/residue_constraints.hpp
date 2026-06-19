// Hand-rolled C++ port of the per-residue amino-acid constraint logic from
// boltzgen.data.parse.schema and boltzgen.model.modules.inverse_fold.
//
// Mirrors: parse_range, _normalize_aa_spec, _convert_aa_names_to_indices,
// parse_residue_constraints, build_constraint_logit_mask.
//
// Errors are reported by throwing std::invalid_argument with messages that
// contain the same substrings the Python ValueErrors do, so the C++ tests can
// assert on them analogously to pytest.raises(match=...).
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace boltz {

// --- parse_range ------------------------------------------------------------
// Parse a 1-indexed residue range spec ("3", "3..5", "..5", "3..", or
// comma-separated combinations) into 0-indexed chain positions.
// c_end == nullopt means open-ended chains are not allowed (matches c_end=None;
// the "N.." form requires c_end). Throws on malformed/out-of-bounds specs.
std::vector<int> parse_range(const std::string& ranges, int c_start = 0,
                             std::optional<int> c_end = std::nullopt);

// --- amino-acid spec --------------------------------------------------------
// An amino-acid specification, mirroring the Python union that
// _normalize_aa_spec accepts: a string ("AGS" / "ALA"), a list (["A","GLY"]),
// or an invalid type (to exercise the error path).
struct AaSpec {
    enum class Kind { String, List, Invalid } kind = Kind::String;
    std::string str;                   // when kind == String
    std::vector<std::string> list;     // when kind == List

    static AaSpec of_string(std::string s) {
        AaSpec a; a.kind = Kind::String; a.str = std::move(s); return a;
    }
    static AaSpec of_list(std::vector<std::string> l) {
        AaSpec a; a.kind = Kind::List; a.list = std::move(l); return a;
    }
    static AaSpec invalid() { AaSpec a; a.kind = Kind::Invalid; return a; }
};

std::vector<std::string> normalize_aa_spec(const AaSpec& spec);

std::vector<int> convert_aa_names_to_indices(const std::vector<std::string>& names);

// --- parse_residue_constraints ---------------------------------------------
// One YAML constraint entry. `position` is stringified by the caller (Python
// does str(position_spec)); has_position=false models a missing key.
struct ConstraintSpec {
    bool has_position = false;
    std::string position;
    std::optional<AaSpec> allowed;
    std::optional<AaSpec> disallowed;
};

// Row-major (chain_length x 20) mask: 0.0 = allowed, 1.0 = disallowed.
struct ConstraintMask {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;  // size rows*cols
    float at(int r, int c) const { return data[r * cols + c]; }
    float& at(int r, int c) { return data[r * cols + c]; }
    float row_sum(int r) const;
};

ConstraintMask parse_residue_constraints(const std::vector<ConstraintSpec>& specs,
                                         int chain_length);

// --- build_constraint_logit_mask -------------------------------------------
// Optional per-residue mask input (num_nodes x num_aa). present=false models
// aa_constraint_mask=None. rows/cols capture the actual shape so the
// shape-mismatch path can be exercised.
struct AaConstraintMaskInput {
    bool present = false;
    int rows = 0;
    int cols = 0;
    std::vector<float> data;  // size rows*cols
};

// Output additive logit bias (num_nodes x num_aa): 0.0 = allowed, -inf_val =
// disallowed. Warnings are collected rather than emitted, so tests can assert
// on them like pytest.warns.
struct LogitMaskResult {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;  // size rows*cols
    std::vector<std::string> warnings;
    float at(int r, int c) const { return data[r * cols + c]; }
};

LogitMaskResult build_constraint_logit_mask(
    int num_nodes, const AaConstraintMaskInput& aa_constraint_mask,
    const std::vector<std::string>& inverse_fold_restriction, float inf_val);

}  // namespace boltz
