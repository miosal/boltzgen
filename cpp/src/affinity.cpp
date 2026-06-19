#include "boltz/affinity.hpp"

#include <algorithm>

namespace boltz {
namespace {

std::vector<float> apply(const std::vector<float>& x, const LinearB& L, bool relu) {
    std::vector<float> y(L.out);
    for (int o = 0; o < L.out; ++o) {
        float s = L.b[o];
        for (int i = 0; i < L.in; ++i) s += L.w[o * L.in + i] * x[i];
        y[o] = relu ? std::max(0.0f, s) : s;
    }
    return y;
}

}  // namespace

AffinityOutputs affinity_head(const std::vector<float>& z, const std::vector<float>& lig_mask,
                              const std::vector<float>& rec_mask, int N,
                              const AffinityWeights& w) {
    const int tz = w.token_z;

    // Cross-pair mask: ligand-receptor, receptor-ligand, ligand-ligand; no diagonal.
    std::vector<float> g(tz, 0.0f);
    float denom = 0.0f;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            const float cpm = lig_mask[i] * rec_mask[j] + rec_mask[i] * lig_mask[j] +
                              lig_mask[i] * lig_mask[j];
            if (cpm == 0.0f) continue;
            for (int c = 0; c < tz; ++c) g[c] += z[(i * N + j) * tz + c] * cpm;
            denom += cpm;
        }
    for (int c = 0; c < tz; ++c) g[c] /= (denom + 1e-7f);

    // affinity_out_mlp: (tz->tz) relu (tz->ts) relu
    auto h = apply(g, w.out_l1, true);
    h = apply(h, w.out_l2, true);

    auto value = apply(apply(apply(h, w.val_l1, true), w.val_l2, true), w.val_l3, false);
    auto score = apply(apply(apply(h, w.sco_l1, true), w.sco_l2, true), w.sco_l3, false);
    auto binary = apply(score, w.binary, false);

    AffinityOutputs out;
    out.pred_value = value[0];
    out.pred_score = score[0];
    out.logits_binary = binary[0];
    return out;
}

}  // namespace boltz
