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
