// End-to-end demonstration that the assembled diffusion module RUNS: a score
// network (atom embed -> + Fourier time -> DiffusionTransformer -> atom decode)
// wrapped in Karras preconditioning and driven by the reverse-diffusion sampler,
// producing coordinates from noise.
//
// NOTE: the atom encoder/decoder here are linear placeholders, not the windowed
// AtomTransformer of the reference model, and weights are synthetic — this
// validates that the diffusion *pipeline* assembles and executes, analogous to
// the inverse-folding example. Faithful atom attention + real weights are the
// remaining integration step.
#include "boltz/diffusion_blocks.hpp"
#include "boltz/diffusion_sampler.hpp"
#include "boltz/diffusion_schedule.hpp"
#include "boltz/diffusion_transformer.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

void fill(std::vector<float>& v, int n, float seed) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.05f * std::sin(seed + 0.21f * i);
}

DiffusionTransformerLayerWeights mk_layer(int dim, int dc, int H) {
    DiffusionTransformerLayerWeights w; w.dim = dim; w.dim_cond = dc; w.num_heads = H;
    w.adaln.dim = dim; w.adaln.dim_cond = dc;
    fill(w.adaln.s_norm_w, dc, 1); fill(w.adaln.s_scale_w, dim * dc, 2);
    fill(w.adaln.s_scale_b, dim, 3); fill(w.adaln.s_bias_w, dim * dc, 4);
    w.attn.c_s = dim; w.attn.c_z = 0; w.attn.num_heads = H; w.attn.compute_pair_bias = false; w.attn.inf = 1e6f;
    fill(w.attn.q_w, dim * dim, 5); fill(w.attn.q_b, dim, 5.5f); fill(w.attn.k_w, dim * dim, 6);
    fill(w.attn.v_w, dim * dim, 7); fill(w.attn.g_w, dim * dim, 8); fill(w.attn.o_w, dim * dim, 9);
    fill(w.op_w, dim * dc, 10); fill(w.op_b, dim, 11);
    w.transition.dim = dim; w.transition.dim_cond = dc; w.transition.dim_inner = dim * 2;
    w.transition.adaln = w.adaln;
    fill(w.transition.swish_gate_w, 2 * (dim * 2) * dim, 12);
    fill(w.transition.a_to_b_w, (dim * 2) * dim, 13);
    fill(w.transition.b_to_a_w, dim * (dim * 2), 14);
    fill(w.transition.op_w, dim * dc, 15); fill(w.transition.op_b, dim, 16);
    return w;
}

}  // namespace

BOLTZ_TEST(diffusion_pipeline_runs_and_is_deterministic) {
    const int M = 4, dim = 4, dc = 4, H = 2;
    DiffusionSchedule sched(DiffusionScheduleConfig{});
    auto sigmas = sched.sample_schedule_af3(8);

    // Score-net components (synthetic).
    std::vector<float> atom_embed(dim * 3), atom_decode(3 * dim), four_w(dim), four_b(dim);
    fill(atom_embed, dim * 3, 100); fill(atom_decode, 3 * dim, 200);
    fill(four_w, dim, 300); fill(four_b, dim, 400);
    std::vector<float> s_cond(M * dc, 0.05f);  // fixed conditioning
    std::vector<float> tmask(M, 1.0f), bias0(M * M * H, 0.0f);
    std::vector<DiffusionTransformerLayerWeights> layers = {mk_layer(dim, dc, H)};

    // Raw score model: scaled coords [M*3] + time -> coordinate update [M*3].
    auto net = [&](const std::vector<float>& scaled, double t) {
        auto te = fourier_embedding(static_cast<float>(t), four_w, four_b);  // [dim]
        std::vector<float> a(M * dim);
        for (int i = 0; i < M; ++i)
            for (int d = 0; d < dim; ++d) {
                float v = te[d];
                for (int c = 0; c < 3; ++c) v += atom_embed[d * 3 + c] * scaled[i * 3 + c];
                a[i * dim + d] = v;
            }
        std::vector<std::vector<float>> biases = {bias0};
        auto a_out = diffusion_transformer(a, s_cond, biases, tmask, M, layers);
        std::vector<float> r(M * 3);
        for (int i = 0; i < M; ++i)
            for (int c = 0; c < 3; ++c) {
                float v = 0;
                for (int d = 0; d < dim; ++d) v += atom_decode[c * dim + d] * a_out[i * dim + d];
                r[i * 3 + c] = v;
            }
        return r;
    };

    auto denoise = [&](const std::vector<float>& coords, double t_hat) {
        return preconditioned_forward(coords, t_hat, sched, net);
    };

    auto run_once = [&]() {
        unsigned st = 777;
        auto randn = [&]() { st = st * 1664525u + 1013904223u; return static_cast<float>(st >> 9) / 8388608.0f - 1.0f; };
        return sample_diffusion(sigmas, M, tmask, denoise, randn, SamplerParams{});
    };

    auto a = run_once();
    auto b = run_once();
    expect_eq_i(static_cast<long>(a.size()), M * 3, "coords shape");
    for (float v : a) expect_true(std::isfinite(v), "finite coords");
    for (size_t i = 0; i < a.size(); ++i) expect_eq_f(a[i], b[i], "deterministic");
}

int main() {
    std::printf("== test_diffusion_pipeline ==\n");
    return boltztest::run_all();
}
