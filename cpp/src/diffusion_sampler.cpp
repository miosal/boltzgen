#include "boltz/diffusion_sampler.hpp"

#include <cmath>

namespace boltz {
namespace {

// Subtract the mask-weighted centroid (per the `center` util).
void center(std::vector<float>& coords, int M, const std::vector<float>& mask) {
    double cx = 0, cy = 0, cz = 0, n = 0;
    for (int i = 0; i < M; ++i) {
        const float m = mask[i];
        cx += m * coords[i * 3 + 0];
        cy += m * coords[i * 3 + 1];
        cz += m * coords[i * 3 + 2];
        n += m;
    }
    if (n < 1.0) n = 1.0;
    cx /= n; cy /= n; cz /= n;
    for (int i = 0; i < M; ++i) {
        coords[i * 3 + 0] -= static_cast<float>(cx);
        coords[i * 3 + 1] -= static_cast<float>(cy);
        coords[i * 3 + 2] -= static_cast<float>(cz);
    }
}

}  // namespace

std::vector<float> preconditioned_forward(
    const std::vector<float>& coords_noisy, double sigma, const DiffusionSchedule& sched,
    const std::function<std::vector<float>(const std::vector<float>&, double)>& net) {
    const double c_in = sched.c_in(sigma);
    const double c_skip = sched.c_skip(sigma);
    const double c_out = sched.c_out(sigma);
    const double t = sched.c_noise(sigma);

    std::vector<float> scaled(coords_noisy.size());
    for (size_t i = 0; i < scaled.size(); ++i) scaled[i] = static_cast<float>(c_in * coords_noisy[i]);

    std::vector<float> r = net(scaled, t);
    std::vector<float> denoised(coords_noisy.size());
    for (size_t i = 0; i < denoised.size(); ++i)
        denoised[i] = static_cast<float>(c_skip * coords_noisy[i] + c_out * r[i]);
    return denoised;
}

std::vector<float> sample_diffusion(
    const std::vector<double>& sigmas, int M, const std::vector<float>& mask,
    const std::function<std::vector<float>(const std::vector<float>&, double)>& denoise,
    const std::function<float()>& randn, const SamplerParams& params) {
    const int M3 = M * 3;

    // atom positions start as noise at init_sigma.
    std::vector<float> coords(M3);
    const double init_sigma = sigmas.front();
    for (int i = 0; i < M3; ++i) coords[i] = static_cast<float>(init_sigma * randn());

    const int n = static_cast<int>(sigmas.size()) - 1;  // number of steps
    for (int step = 0; step < n; ++step) {
        const double sigma_tm = sigmas[step];
        const double sigma_t = sigmas[step + 1];
        const double gamma = (sigma_tm > params.gamma_min) ? params.gamma_0 : 0.0;
        const double t_hat = sigma_tm * (1.0 + gamma);
        const double noise_var = params.noise_scale * params.noise_scale *
                                 (t_hat * t_hat - sigma_tm * sigma_tm);

        center(coords, M, mask);

        // Add noise.
        std::vector<float> noisy(M3);
        const double sd = params.noise_scale * std::sqrt(noise_var > 0 ? noise_var : 0.0);
        for (int i = 0; i < M3; ++i) noisy[i] = coords[i] + static_cast<float>(sd * randn());

        // Denoise via the score network.
        std::vector<float> denoised = denoise(noisy, t_hat);

        // ODE step.
        for (int i = 0; i < M3; ++i) {
            const double dos = (noisy[i] - denoised[i]) / t_hat;
            coords[i] = static_cast<float>(noisy[i] + params.step_scale * (sigma_t - t_hat) * dos);
        }
    }
    return coords;
}

}  // namespace boltz
