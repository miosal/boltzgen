// Tests for hand-rolled featurization helpers. Expected values derived directly
// from the GaussianSmearing formula in boltzgen.model.modules.inverse_fold.
#include "boltz/featurizer.hpp"
#include "test_framework.hpp"

#include <cmath>

using namespace boltz;
using namespace boltztest;

BOLTZ_TEST(gaussian_smearing_shape) {
    auto m = gaussian_smearing({0.0f, 1.0f, 2.0f}, 0.0f, 5.0f, 6);
    expect_eq_i(m.rows, 3, "rows = n_dist");
    expect_eq_i(m.cols, 6, "cols = n_gaussians");
}

BOLTZ_TEST(gaussian_smearing_values) {
    // offsets = [0,1,2,3,4,5], delta = 1, coeff = -0.5.
    auto m = gaussian_smearing({2.0f}, 0.0f, 5.0f, 6);
    const float coeff = -0.5f;
    for (int k = 0; k < 6; ++k) {
        float d = 2.0f - static_cast<float>(k);
        expect_eq_f(m.at(0, k), std::exp(coeff * d * d), "smear value");
    }
    // Peak (=1.0) exactly at the offset equal to the distance.
    expect_eq_f(m.at(0, 2), 1.0f, "peak at offset==dist");
}

BOLTZ_TEST(gaussian_smearing_16_bins_range20) {
    // Inverse-folding default: 16 gaussians over 0..20 A.
    auto m = gaussian_smearing({0.0f}, 0.0f, 20.0f, 16);
    expect_eq_i(m.cols, 16, "16 bins");
    expect_eq_f(m.at(0, 0), 1.0f, "dist 0 peaks at first offset (0)");
}

int main() {
    std::printf("== test_featurizer ==\n");
    return boltztest::run_all();
}
