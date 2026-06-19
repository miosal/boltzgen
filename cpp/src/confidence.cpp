#include "boltz/confidence.hpp"

#include <cmath>

namespace boltz {
namespace {

void linear_nobias(const float* x, int in, int out, const std::vector<float>& W, float* y) {
    for (int o = 0; o < out; ++o) {
        float s = 0;
        for (int i = 0; i < in; ++i) s += W[o * in + i] * x[i];
        y[o] = s;
    }
}

}  // namespace

ConfidenceOutputs confidence_heads(const std::vector<float>& s, const std::vector<float>& z,
                                   int N, const ConfidenceWeights& w) {
    const int ts = w.token_s, tz = w.token_z;
    ConfidenceOutputs out;
    out.pae_logits.resize((size_t)N * N * w.num_pae_bins);
    out.pde_logits.resize((size_t)N * N * w.num_pde_bins);
    out.plddt_logits.resize((size_t)N * w.num_plddt_bins);
    out.resolved_logits.resize((size_t)N * 2);

    // Pair heads.
    std::vector<float> zsym(tz);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const float* zij = &z[(i * N + j) * tz];
            linear_nobias(zij, tz, w.num_pae_bins, w.to_pae, &out.pae_logits[(i * N + j) * w.num_pae_bins]);
            // PDE uses the symmetrized pair rep z + z^T.
            for (int c = 0; c < tz; ++c) zsym[c] = z[(i * N + j) * tz + c] + z[(j * N + i) * tz + c];
            linear_nobias(zsym.data(), tz, w.num_pde_bins, w.to_pde, &out.pde_logits[(i * N + j) * w.num_pde_bins]);
        }

    // Single heads.
    for (int i = 0; i < N; ++i) {
        linear_nobias(&s[i * ts], ts, w.num_plddt_bins, w.to_plddt, &out.plddt_logits[i * w.num_plddt_bins]);
        linear_nobias(&s[i * ts], ts, 2, w.to_resolved, &out.resolved_logits[i * 2]);
    }
    return out;
}

std::vector<float> aggregated_metric(const std::vector<float>& logits, int rows,
                                     int num_bins, float end) {
    const float bin_width = end / num_bins;
    std::vector<float> centers(num_bins);
    for (int b = 0; b < num_bins; ++b) centers[b] = 0.5f * bin_width + b * bin_width;

    std::vector<float> out(rows);
    for (int r = 0; r < rows; ++r) {
        const float* lg = &logits[r * num_bins];
        float mx = lg[0];
        for (int b = 1; b < num_bins; ++b) mx = std::max(mx, lg[b]);
        float sum = 0;
        std::vector<float> p(num_bins);
        for (int b = 0; b < num_bins; ++b) { p[b] = std::exp(lg[b] - mx); sum += p[b]; }
        float val = 0;
        for (int b = 0; b < num_bins; ++b) val += (p[b] / sum) * centers[b];
        out[r] = val;
    }
    return out;
}

}  // namespace boltz
