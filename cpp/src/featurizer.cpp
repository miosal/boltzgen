#include "boltz/featurizer.hpp"

#include <cmath>

namespace boltz {

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
