// EDM/AF3 reverse-diffusion sampler loop (boltzgen.model.modules.diffusion
// AtomDiffusion.sample). Generic over the score network: the caller supplies a
// `denoise` callback (the preconditioned network forward) and a `randn` source,
// so the integration is testable independently of the network. Single example
// (B=1); coordinates are [M, 3] row-major.
#pragma once

#include <functional>
#include <vector>

namespace boltz {

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
