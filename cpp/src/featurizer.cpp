#include "boltz/featurizer.hpp"

#include "boltz/const.hpp"

#include <cmath>

namespace boltz {

TokenFeatures token_features(const std::vector<std::string>& comp_ids,
                             const std::vector<int>& seq_ids) {
    const int n = static_cast<int>(comp_ids.size());
    const int T = num_tokens();
    TokenFeatures f;
    f.n = n;
    f.num_token_types = T;
    f.res_type.assign((size_t)n * T, 0.0f);
    f.token_index.resize(n);
    f.residue_index.resize(n);
    f.mol_type.resize(n);
    f.pad_mask.assign(n, 1.0f);
    for (int i = 0; i < n; ++i) {
        const int tid = token_id(comp_ids[i]);
        f.res_type[(size_t)i * T + tid] = 1.0f;
        f.token_index[i] = i;
        f.residue_index[i] = seq_ids[i];
        f.mol_type[i] = mol_type_of(comp_ids[i]);
    }
    return f;
}

MsaFeatures single_sequence_msa_features(const std::vector<float>& res_type, int n,
                                         int num_token_types) {
    MsaFeatures m;
    m.n = n;
    m.num_token_types = num_token_types;
    m.depth = 1;
    m.msa = res_type;       // one row == the query one-hot
    m.profile = res_type;   // single-sequence profile is the query one-hot
    m.has_deletion.assign(n, 0.0f);
    m.deletion_value.assign(n, 0.0f);
    m.msa_mask.assign(n, 1.0f);
    return m;
}

MsaFeatures msa_features_from_alignment(const std::vector<std::vector<int>>& seq_tokens,
                                        const std::vector<std::vector<float>>& deletions,
                                        int n, int T) {
    const int depth = static_cast<int>(seq_tokens.size());
    MsaFeatures m;
    m.n = n;
    m.num_token_types = T;
    m.depth = depth;
    m.msa.assign((size_t)depth * n * T, 0.0f);
    m.has_deletion.assign((size_t)depth * n, 0.0f);
    m.deletion_value.assign((size_t)depth * n, 0.0f);
    m.msa_mask.assign((size_t)depth * n, 1.0f);
    m.profile.assign((size_t)n * T, 0.0f);

    m.deletion_mean.assign(n, 0.0f);
    for (int d = 0; d < depth; ++d) {
        for (int c = 0; c < n; ++c) {
            const int tok = seq_tokens[d][c];
            m.msa[((size_t)d * n + c) * T + tok] = 1.0f;
            m.profile[(size_t)c * T + tok] += 1.0f;  // accumulate counts
            const float del = deletions[d][c];
            m.has_deletion[(size_t)d * n + c] = del > 0 ? 1.0f : 0.0f;
            m.deletion_value[(size_t)d * n + c] = del;
            m.deletion_mean[c] += del;
        }
    }
    // Normalize profile to column frequencies and deletion to per-column mean.
    for (int c = 0; c < n; ++c) {
        for (int t = 0; t < T; ++t) m.profile[(size_t)c * T + t] /= depth;
        m.deletion_mean[c] /= depth;
    }
    return m;
}

Matrix gaussian_smearing(const std::vector<float>& dist, float start, float stop,
                         int num_gaussians) {
    std::vector<float> offsets(num_gaussians);
    // torch.linspace(start, stop, n): inclusive of both ends.
    const float step = num_gaussians > 1 ? (stop - start) / (num_gaussians - 1) : 0.0f;
    for (int k = 0; k < num_gaussians; ++k) offsets[k] = start + step * k;

    const float delta = num_gaussians > 1 ? (offsets[1] - offsets[0]) : 1.0f;
    const float coeff = -0.5f / (delta * delta);

    Matrix out;
    out.rows = static_cast<int>(dist.size());
    out.cols = num_gaussians;
    out.data.resize(static_cast<size_t>(out.rows) * num_gaussians);
    for (int i = 0; i < out.rows; ++i) {
        for (int k = 0; k < num_gaussians; ++k) {
            const float d = dist[i] - offsets[k];
            out.at(i, k) = std::exp(coeff * d * d);
        }
    }
    return out;
}

}  // namespace boltz
