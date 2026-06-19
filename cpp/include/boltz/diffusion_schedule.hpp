// Hand-rolled port of the EDM/AF3 diffusion noise schedule and Karras
// preconditioning coefficients from boltzgen.model.modules.diffusion
// (AtomDiffusion: c_skip/c_out/c_in/c_noise, sample_schedule_af3,
// sample_schedule_dilated). Pure scalar math — runs on the CPU control path of
// the sampler; the network forward is the only GPU work.
#pragma once

#include <vector>

namespace boltz {

struct DiffusionScheduleConfig {
    double sigma_min = 0.0004;
    double sigma_max = 160.0;
    double sigma_data = 16.0;
    double rho = 7.0;
    int num_sampling_steps = 200;
    // Dilated-schedule params (design.yaml defaults).
    double time_dilation = 2.667;
    double time_dilation_start = 0.6;
    double time_dilation_end = 0.8;
};

class DiffusionSchedule {
public:
    explicit DiffusionSchedule(DiffusionScheduleConfig cfg) : cfg_(cfg) {}

    double c_skip(double sigma) const;
    double c_out(double sigma) const;
    double c_in(double sigma) const;
    double c_noise(double sigma) const;

    // Both return num_steps+1 sigmas, the last padded to 0.0.
    std::vector<double> sample_schedule_af3(int num_sampling_steps = -1) const;
    std::vector<double> sample_schedule_dilated(int num_sampling_steps = -1) const;

    const DiffusionScheduleConfig& config() const { return cfg_; }

private:
    DiffusionScheduleConfig cfg_;
};

}  // namespace boltz
