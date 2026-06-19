// Tests for the diffusion noise schedule + Karras preconditioning. There is no
// Python unit test for this; the expected values are derived analytically from
// the formulas in boltzgen.model.modules.diffusion (AtomDiffusion).
#include "boltz/diffusion_schedule.hpp"
#include "test_framework.hpp"

#include <cmath>

using namespace boltz;
using namespace boltztest;

static void near(double got, double want, double tol, const std::string& what) {
    if (std::fabs(got - want) > tol)
        throw AssertFailure(what + " — got " + std::to_string(got) + " want " +
                            std::to_string(want));
}

static DiffusionSchedule make() {
    DiffusionScheduleConfig c;  // design.yaml defaults
    return DiffusionSchedule(c);
}

// ---- preconditioning coefficients -----------------------------------------
BOLTZ_TEST(precond_at_sigma_data) {
    auto d = make();
    const double sd = d.config().sigma_data;  // 16
    // At sigma == sigma_data the Karras coefficients have closed forms.
    near(d.c_skip(sd), 0.5, 1e-12, "c_skip");
    near(d.c_out(sd), sd / std::sqrt(2.0), 1e-9, "c_out");
    near(d.c_in(sd), 1.0 / (sd * std::sqrt(2.0)), 1e-12, "c_in");
    near(d.c_noise(sd), 0.0, 1e-12, "c_noise");  // log(1)*0.25
}

BOLTZ_TEST(precond_skip_in_limits) {
    auto d = make();
    // sigma -> 0: c_skip -> 1, c_out -> 0.
    near(d.c_skip(1e-9), 1.0, 1e-6, "c_skip->1");
    near(d.c_out(1e-9), 0.0, 1e-6, "c_out->0");
    // Large sigma: c_skip -> 0.
    near(d.c_skip(1e6), 0.0, 1e-6, "c_skip->0");
}

// ---- AF3 schedule ----------------------------------------------------------
BOLTZ_TEST(af3_length_and_padding) {
    auto d = make();
    auto s = d.sample_schedule_af3(200);
    expect_eq_i(static_cast<long>(s.size()), 201, "n+1 length");
    near(s.back(), 0.0, 0.0, "last is 0");
}

BOLTZ_TEST(af3_endpoints) {
    auto d = make();
    const auto& c = d.config();
    auto s = d.sample_schedule_af3(200);
    // step 0 -> sigma_max * sigma_data ; step n-1 -> sigma_min * sigma_data.
    near(s.front(), c.sigma_max * c.sigma_data, 1e-6, "first = smax*sd");
    near(s[s.size() - 2], c.sigma_min * c.sigma_data, 1e-9, "penultimate = smin*sd");
}

BOLTZ_TEST(af3_monotonic_decreasing) {
    auto d = make();
    auto s = d.sample_schedule_af3(200);
    for (size_t i = 1; i < s.size(); ++i)
        expect_true(s[i] <= s[i - 1] + 1e-12, "non-increasing");
}

// ---- dilated schedule ------------------------------------------------------
BOLTZ_TEST(dilated_endpoints_match_af3) {
    auto d = make();
    const auto& c = d.config();
    auto s = d.sample_schedule_dilated(200);
    expect_eq_i(static_cast<long>(s.size()), 201, "n+1 length");
    // dilate maps 0->0 and 1->1, so endpoints equal the AF3 endpoints.
    near(s.front(), c.sigma_max * c.sigma_data, 1e-6, "first = smax*sd");
    near(s[s.size() - 2], c.sigma_min * c.sigma_data, 1e-9, "penultimate = smin*sd");
    near(s.back(), 0.0, 0.0, "last is 0");
}

BOLTZ_TEST(dilated_monotonic_decreasing) {
    auto d = make();
    auto s = d.sample_schedule_dilated(200);
    for (size_t i = 1; i < s.size(); ++i)
        expect_true(s[i] <= s[i - 1] + 1e-9, "non-increasing");
}

int main() {
    std::printf("== test_diffusion_schedule ==\n");
    return boltztest::run_all();
}
