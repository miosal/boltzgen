// Tests for the EDM/AF3 reverse-diffusion sampler loop.
#include "boltz/diffusion_sampler.hpp"
#include "boltz/diffusion_schedule.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

// With a constant (oracle) denoiser that always returns a fixed zero-mean target
// x0, zero injected noise (noise_scale=0) and gamma_0=0 (t_hat = sigma_tm), the
// final step has sigma_t = 0 and lands exactly on x0:
//   coords_next = noisy + (0 - t_hat)*(noisy - x0)/t_hat = x0.
BOLTZ_TEST(sampler_converges_to_oracle_target) {
    const int M = 4;
    // Zero-mean target so per-step centering is a no-op on x0.
    std::vector<float> x0 = {1, 0, 0,  -1, 0, 0,  0, 2, 0,  0, -2, 0};
    std::vector<float> mask(M, 1.0f);

    DiffusionScheduleConfig cfg;
    DiffusionSchedule sched(cfg);
    auto sigmas = sched.sample_schedule_af3(20);

    auto denoise = [&](const std::vector<float>&, double) { return x0; };
    auto randn = []() { return 0.0f; };  // deterministic, no noise

    SamplerParams p;
    p.gamma_0 = 0.0;      // t_hat = sigma_tm
    p.noise_scale = 0.0;  // no injected noise
    p.step_scale = 1.0;

    auto coords = sample_diffusion(sigmas, M, mask, denoise, randn, p);
    expect_eq_i(static_cast<long>(coords.size()), M * 3, "shape");
    for (int i = 0; i < M * 3; ++i)
        if (std::fabs(coords[i] - x0[i]) > 1e-3f)
            throw AssertFailure("did not converge to oracle target at " + std::to_string(i));
}

BOLTZ_TEST(sampler_runs_with_noise_and_finite) {
    const int M = 3;
    std::vector<float> mask(M, 1.0f);
    DiffusionSchedule sched(DiffusionScheduleConfig{});
    auto sigmas = sched.sample_schedule_af3(10);

    // A denoiser that pulls toward the origin (shrinks coords).
    auto denoise = [&](const std::vector<float>& c, double) {
        std::vector<float> d(c.size());
        for (size_t i = 0; i < c.size(); ++i) d[i] = 0.5f * c[i];
        return d;
    };
    // Deterministic pseudo-noise.
    unsigned state = 12345;
    auto randn = [&]() {
        state = state * 1664525u + 1013904223u;
        return (static_cast<float>(state >> 9) / 8388608.0f - 1.0f);  // ~[-1,1]
    };

    SamplerParams p;
    auto coords = sample_diffusion(sigmas, M, mask, denoise, randn, p);
    expect_eq_i(static_cast<long>(coords.size()), M * 3, "shape");
    for (float v : coords) expect_true(std::isfinite(v), "finite output");
}

int main() {
    std::printf("== test_diffusion_sampler ==\n");
    return boltztest::run_all();
}
