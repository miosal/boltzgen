#include "boltz/residue_constraints.hpp"

#include "boltz/const.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace boltz {
namespace {

std::string strip_upper(const std::string& in) {
    size_t b = 0, e = in.size();
    while (b < e && std::isspace(static_cast<unsigned char>(in[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(in[e - 1]))) --e;
    std::string out = in.substr(b, e - b);
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

bool all_digits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

bool all_alpha(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!std::isalpha(static_cast<unsigned char>(c))) return false;
    return true;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream ss(s);
    while (std::getline(ss, cur, sep)) out.push_back(cur);
    return out;
}

std::vector<char> chars_of(const std::string& s) {
    return std::vector<char>(s.begin(), s.end());
}

}  // namespace

float ConstraintMask::row_sum(int r) const {
    float s = 0.0f;
    for (int c = 0; c < cols; ++c) s += at(r, c);
    return s;
}

std::vector<int> parse_range(const std::string& ranges, int c_start,
                             std::optional<int> c_end) {
    std::vector<std::string> spec_list =
        ranges.find(',') != std::string::npos ? split(ranges, ',')
                                              : std::vector<std::string>{ranges};
    std::vector<int> indices;
    for (const std::string& spec : spec_list) {
        int start = 0, end = 0;
        const size_t dots = spec.find("..");
        if (all_digits(spec)) {
            // Single number: 1-indexed -> 0-indexed.
            start = std::stoi(spec) - 1;
            end = std::stoi(spec) - 1;
            indices.push_back(c_start + start);
        } else if (dots != std::string::npos) {
            const std::string left = spec.substr(0, dots);
            const std::string right = spec.substr(dots + 2);
            if (all_digits(left) && all_digits(right)) {
                // "a..b": inclusive end (1-indexed end == 0-indexed exclusive).
                start = std::stoi(left) - 1;
                end = std::stoi(right);
                for (int i = c_start + start; i < c_start + end; ++i) indices.push_back(i);
            } else if (left.empty() && all_digits(right)) {
                // "..b": from start, inclusive of b (1-indexed).
                end = std::stoi(right);
                start = 0;
                for (int i = c_start; i < c_start + end; ++i) indices.push_back(i);
            } else if (all_digits(left) && right.empty()) {
                // "a..": from a (1-indexed) to end of chain; requires c_end.
                if (!c_end.has_value())
                    throw std::invalid_argument("open-ended range requires chain length");
                start = std::stoi(left) - 1;
                end = *c_end - c_start;
                for (int i = c_start + start; i < *c_end; ++i) indices.push_back(i);
            } else {
                throw std::invalid_argument("Malformed residue range specification '" +
                                            spec + "' in '" + ranges + "'.");
            }
        } else {
            throw std::invalid_argument("Malformed residue range specification '" +
                                        spec + "' in '" + ranges + "'.");
        }

        if (start < 0) {
            throw std::invalid_argument("There is a 0 in the specified range(s) " +
                                        ranges + ". Residue indices are 1 indexed.");
        }
        if (c_end.has_value() && end > *c_end - c_start) {
            throw std::invalid_argument("Specified end " + ranges +
                                        " is higher than the length of the chain.");
        }
    }
    return indices;
}

std::vector<std::string> normalize_aa_spec(const AaSpec& spec) {
    if (spec.kind == AaSpec::Kind::String) {
        const std::string s = strip_upper(spec.str);
        std::vector<std::string> out;
        if (s.size() <= 3 && all_alpha(s)) {
            if (s.size() == 3 && canonical_index(s) != -1) {
                out.push_back(s);
                return out;
            }
            for (char c : chars_of(s)) out.emplace_back(1, c);
            return out;
        }
        for (char c : chars_of(s)) out.emplace_back(1, c);
        return out;
    }
    if (spec.kind == AaSpec::Kind::List) {
        std::vector<std::string> out;
        out.reserve(spec.list.size());
        for (const std::string& x : spec.list) out.push_back(strip_upper(x));
        return out;
    }
    throw std::invalid_argument("Invalid amino acid specification");
}

std::vector<int> convert_aa_names_to_indices(const std::vector<std::string>& names) {
    const auto& letter_map = prot_letter_to_token();
    std::vector<int> indices;
    for (std::string name : names) {
        name = strip_upper(name);
        if (name.size() == 1) {
            auto it = letter_map.find(name);
            if (it == letter_map.end())
                throw std::invalid_argument("Unknown amino acid code: " + name);
            name = it->second;
        }
        const int idx = canonical_index(name);
        if (idx == -1) throw std::invalid_argument("Unknown amino acid: " + name);
        indices.push_back(idx);
    }
    return indices;
}

ConstraintMask parse_residue_constraints(const std::vector<ConstraintSpec>& specs,
                                         int chain_length) {
    const int num_aa = static_cast<int>(canonical_tokens().size());  // 20
    ConstraintMask mask;
    mask.rows = chain_length;
    mask.cols = num_aa;
    mask.data.assign(static_cast<size_t>(chain_length) * num_aa, 0.0f);

    for (const ConstraintSpec& c : specs) {
        if (!c.has_position)
            throw std::invalid_argument("residue_constraints: 'position' is required");

        const std::vector<int> positions = parse_range(c.position, 0, chain_length);
        for (int pos : positions) {
            if (pos < 0 || pos >= chain_length) {
                throw std::invalid_argument(
                    "Position " + std::to_string(pos + 1) +
                    " is out of bounds for chain of length " + std::to_string(chain_length));
            }
        }

        const bool has_allowed = c.allowed.has_value();
        const bool has_disallowed = c.disallowed.has_value();
        if (has_allowed && has_disallowed) {
            throw std::invalid_argument("Position " + c.position +
                                        ": cannot specify both 'allowed' and 'disallowed'");
        }
        if (!has_allowed && !has_disallowed) {
            throw std::invalid_argument("Position " + c.position +
                                        ": must specify either 'allowed' or 'disallowed'");
        }

        if (has_allowed) {
            const std::vector<std::string> aa_list = normalize_aa_spec(*c.allowed);
            if (aa_list.empty())
                throw std::invalid_argument("Position " + c.position + ": 'allowed' cannot be empty");
            const std::vector<int> aa_indices = convert_aa_names_to_indices(aa_list);
            std::vector<float> new_block(num_aa, 1.0f);
            for (int idx : aa_indices) new_block[idx] = 0.0f;
            for (int pos : positions) {
                for (int j = 0; j < num_aa; ++j)
                    mask.at(pos, j) = std::max(mask.at(pos, j), new_block[j]);
            }
        } else {
            const std::vector<std::string> aa_list = normalize_aa_spec(*c.disallowed);
            const std::vector<int> aa_indices = convert_aa_names_to_indices(aa_list);
            for (int pos : positions)
                for (int idx : aa_indices) mask.at(pos, idx) = 1.0f;
        }
    }
    return mask;
}

LogitMaskResult build_constraint_logit_mask(
    int num_nodes, const AaConstraintMaskInput& aa_constraint_mask,
    const std::vector<std::string>& inverse_fold_restriction, float inf_val) {
    const int num_aa = static_cast<int>(canonical_tokens().size());  // 20
    LogitMaskResult res;
    res.rows = num_nodes;
    res.cols = num_aa;

    std::vector<char> per_residue_blocked(static_cast<size_t>(num_nodes) * num_aa, 0);
    bool has_per_residue = false;

    if (aa_constraint_mask.present) {
        if (aa_constraint_mask.rows != num_nodes || aa_constraint_mask.cols != num_aa) {
            res.warnings.push_back(
                "aa_constraint_mask shape mismatch: got (" +
                std::to_string(aa_constraint_mask.rows) + ", " +
                std::to_string(aa_constraint_mask.cols) + "), expected (" +
                std::to_string(num_nodes) + ", " + std::to_string(num_aa) +
                "). Ignoring per-residue constraints.");
        } else {
            has_per_residue = true;
            for (size_t i = 0; i < per_residue_blocked.size(); ++i)
                per_residue_blocked[i] = aa_constraint_mask.data[i] > 0 ? 1 : 0;
        }
    }

    std::vector<char> global_blocked(num_aa, 0);
    for (const std::string& res_type : inverse_fold_restriction) {
        const int idx = canonical_index(res_type);
        if (idx == -1)
            throw std::invalid_argument("'" + res_type + "' is not in canonical_tokens");
        global_blocked[idx] = 1;
    }

    auto combined = [&](int r, int c) -> char {
        return (per_residue_blocked[r * num_aa + c] || global_blocked[c]) ? 1 : 0;
    };
    auto row_all_blocked = [&](int r) -> bool {
        for (int c = 0; c < num_aa; ++c) if (!combined(r, c)) return false;
        return true;
    };

    std::vector<int> all_blocked_rows;
    for (int r = 0; r < num_nodes; ++r) if (row_all_blocked(r)) all_blocked_rows.push_back(r);

    if (!all_blocked_rows.empty() && has_per_residue) {
        std::string positions;
        for (size_t i = 0; i < all_blocked_rows.size(); ++i)
            positions += (i ? ", " : "") + std::to_string(all_blocked_rows[i]);
        res.warnings.push_back(
            "Positions [" + positions +
            "] have all amino acids blocked by the combination of per-residue "
            "constraints and '--inverse_fold_avoid'. Relaxing per-residue "
            "constraints for these positions.");
        for (int r : all_blocked_rows)
            for (int c = 0; c < num_aa; ++c) per_residue_blocked[r * num_aa + c] = 0;
    }

    std::vector<int> still_blocked;
    for (int r = 0; r < num_nodes; ++r) if (row_all_blocked(r)) still_blocked.push_back(r);
    if (!still_blocked.empty()) {
        std::string positions;
        for (size_t i = 0; i < still_blocked.size(); ++i)
            positions += (i ? ", " : "") + std::to_string(still_blocked[i]);
        throw std::invalid_argument(
            "Inverse folding has no valid amino acids at token positions [" + positions +
            "] after applying '--inverse_fold_avoid'. Reduce global restrictions to "
            "keep at least one amino acid.");
    }

    res.data.assign(static_cast<size_t>(num_nodes) * num_aa, 0.0f);
    for (int r = 0; r < num_nodes; ++r)
        for (int c = 0; c < num_aa; ++c)
            res.data[r * num_aa + c] = combined(r, c) ? -inf_val : 0.0f;
    return res;
}

}  // namespace boltz
