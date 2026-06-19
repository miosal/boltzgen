// EDM/AF3 reverse-diffusion sampler loop (boltzgen.model.modules.diffusion
// AtomDiffusion.sample). Generic over the score network: the caller supplies a
// `denoise` callback (the preconditioned network forward) and a `randn` source,
// so the integration is testable independently of the network. Single example
// (B=1); coordinates are [M, 3] row-major.
#pragma once

#include "boltz/diffusion_schedule.hpp"

#include <functional>
#include <vector>

namespace boltz {

// Karras preconditioned network forward (AtomDiffusion.preconditioned_network_forward):
//   denoised = c_skip(sigma)*x_noisy + c_out(sigma) * net(c_in(sigma)*x_noisy, c_noise(sigma))
// `net(scaled_coords, t)` is the raw score model returning the coordinate update.
std::vector<float> preconditioned_forward(
    const std::vector<float>& coords_noisy, double sigma, const DiffusionSchedule& sched,
    const std::function<std::vector<float>(const std::vector<float>&, double)>& net);

struct SamplerParams {
    double gamma_0 = 0.8;
    double gamma_min = 1.0;
    double step_scale = 1.5;
    double noise_scale = 1.0;
};

// sigmas: the noise schedule (length num_steps+1, last == 0), e.g. from
// DiffusionSchedule. mask: [M] (1 = real atom, used for centering). denoise:
// (coords_noisy[M*3], t_hat) -> denoised[M*3]. randn: standard-normal source.
std::vector<float> sample_diffusion(
    const std::vector<double>& sigmas, int M, const std::vector<float>& mask,
    const std::function<std::vector<float>(const std::vector<float>&, double)>& denoise,
    const std::function<float()>& randn, const SamplerParams& params);

}  // namespace boltz
