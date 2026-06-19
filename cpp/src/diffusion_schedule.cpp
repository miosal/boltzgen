#include "boltz/diffusion_schedule.hpp"

#include <cassert>
#include <cmath>

namespace boltz {

double DiffusionSchedule::c_skip(double sigma) const {
    const double sd2 = cfg_.sigma_data * cfg_.sigma_data;
    return sd2 / (sigma * sigma + sd2);
}

double DiffusionSchedule::c_out(double sigma) const {
    const double sd2 = cfg_.sigma_data * cfg_.sigma_data;
    return sigma * cfg_.sigma_data / std::sqrt(sd2 + sigma * sigma);
}

double DiffusionSchedule::c_in(double sigma) const {
    const double sd2 = cfg_.sigma_data * cfg_.sigma_data;
    return 1.0 / std::sqrt(sigma * sigma + sd2);
}

double DiffusionSchedule::c_noise(double sigma) const {
    return std::log(sigma / cfg_.sigma_data) * 0.25;
}

std::vector<double> DiffusionSchedule::sample_schedule_af3(int num_sampling_steps) const {
    const int n = num_sampling_steps > 0 ? num_sampling_steps : cfg_.num_sampling_steps;
    const double inv_rho = 1.0 / cfg_.rho;
    const double smax = std::pow(cfg_.sigma_max, inv_rho);
    const double smin = std::pow(cfg_.sigma_min, inv_rho);

    std::vector<double> sigmas;
    sigmas.reserve(n + 1);
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / (n - 1);
        const double base = smax + t * (smin - smax);
        sigmas.push_back(std::pow(base, cfg_.rho) * cfg_.sigma_data);
    }
    sigmas.push_back(0.0);  // F.pad last step to sigma 0
    return sigmas;
}

namespace {
// Reparameterization of t in [0,1] that dilates [start,end] by `dilation`.
double dilate(double ts, double start, double end, double dilation) {
    const double x = end - start;
    const double l = start;
    const double u = 1.0 - end;
    assert((dilation - 1.0) * x <= l + u && "dilation too large");

    const double inv_dilation = 1.0 / dilation;
    const double ratio = (l + u + (1.0 - dilation) * x) / (l + u);
    const double inv_ratio = 1.0 / ratio;
    const double lprime = l * ratio;
    const double xprime = x * dilation;

    if (ts < lprime) return ts * inv_ratio;
    if (ts < lprime + xprime) return (ts - lprime) * inv_dilation + l;
    return (ts - (lprime + xprime)) * inv_ratio + l + x;
}
}  // namespace

std::vector<double> DiffusionSchedule::sample_schedule_dilated(int num_sampling_steps) const {
    const int n = num_sampling_steps > 0 ? num_sampling_steps : cfg_.num_sampling_steps;
    const double inv_rho = 1.0 / cfg_.rho;
    const double smax = std::pow(cfg_.sigma_max, inv_rho);
    const double smin = std::pow(cfg_.sigma_min, inv_rho);

    std::vector<double> sigmas;
    sigmas.reserve(n + 1);
    for (int i = 0; i < n; ++i) {
        const double ts = static_cast<double>(i) / (n - 1);
        const double dts = dilate(ts, cfg_.time_dilation_start, cfg_.time_dilation_end,
                                  cfg_.time_dilation);
        const double base = smax + dts * (smin - smax);
        sigmas.push_back(std::pow(base, cfg_.rho) * cfg_.sigma_data);
    }
    sigmas.push_back(0.0);
    return sigmas;
}

}  // namespace boltz
