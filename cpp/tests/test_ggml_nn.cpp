// Equivalence tests for the ggml nn primitives: each ggml subgraph is run on
// the CPU backend and checked against an independent scalar reference computed
// in this file. This validates the graph wiring (op choice, weight layout)
// matches the intended PyTorch math, with no model weights required.
#include "boltz/ggml_nn.hpp"
#include "test_framework.hpp"

#include "ggml-cpu.h"

#include <cmath>
#include <vector>

using namespace boltztest;

namespace {

struct Ctx {
    ggml_context* ctx;
    explicit Ctx(size_t mb = 32) {
        ggml_init_params p{mb * 1024 * 1024, nullptr, false};
        ctx = ggml_init(p);
    }
    ~Ctx() { ggml_free(ctx); }
};

void compute(ggml_context* ctx, ggml_tensor* out) {
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS)
        throw AssertFailure("ggml graph compute failed");
}

void set(ggml_tensor* t, const std::vector<float>& v) {
    float* d = ggml_get_data_f32(t);
    for (size_t i = 0; i < v.size(); ++i) d[i] = v[i];
}

float gelu_ref(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

}  // namespace

BOLTZ_TEST(linear_matches_reference) {
    Ctx c;
    const int in = 3, out = 2;
    // W logical [out, in] = [[1,2,3],[4,5,6]] stored ne=[in,out] row-major rows.
    ggml_tensor* W = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, in, out);
    ggml_tensor* b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, out);
    ggml_tensor* x = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, in);
    set(W, {1, 2, 3, 4, 5, 6});
    set(b, {10, 20});
    set(x, {1, 0, -1});

    ggml_tensor* y = boltz::nn::linear(c.ctx, x, W, b);
    compute(c.ctx, y);

    const float* yd = ggml_get_data_f32(y);
    // row0: 1*1+2*0+3*-1 +10 = 8 ; row1: 4-6 +20 = 18
    expect_eq_f(yd[0], 8.0f, "y0");
    expect_eq_f(yd[1], 18.0f, "y1");
}

BOLTZ_TEST(layer_norm_matches_reference) {
    Ctx c;
    const int n = 4;
    ggml_tensor* x = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n);
    ggml_tensor* w = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n);
    ggml_tensor* b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n);
    std::vector<float> xv = {1, 2, 3, 4};
    set(x, xv);
    set(w, {1, 1, 1, 1});
    set(b, {0, 0, 0, 0});

    const float eps = 1e-5f;
    ggml_tensor* y = boltz::nn::layer_norm(c.ctx, x, w, b, eps);
    compute(c.ctx, y);

    // Reference: population mean/var over the feature dim.
    float mean = 0;
    for (float v : xv) mean += v;
    mean /= n;
    float var = 0;
    for (float v : xv) var += (v - mean) * (v - mean);
    var /= n;
    const float* yd = ggml_get_data_f32(y);
    for (int i = 0; i < n; ++i) {
        float ref = (xv[i] - mean) / std::sqrt(var + eps);
        expect_eq_f(yd[i], ref, "ln");
    }
}

BOLTZ_TEST(gelu_erf_matches_reference) {
    Ctx c;
    std::vector<float> xv = {-2, -1, 0, 0.5f, 1, 2, 3};
    ggml_tensor* x = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, (int)xv.size());
    set(x, xv);
    ggml_tensor* y = boltz::nn::gelu(c.ctx, x);
    compute(c.ctx, y);
    const float* yd = ggml_get_data_f32(y);
    for (size_t i = 0; i < xv.size(); ++i)
        expect_eq_f(yd[i], gelu_ref(xv[i]), "gelu");
}

// A 2-layer MLP (Linear -> GELU -> Linear), the shape used inside MLPAttnGNN,
// validated end-to-end against a scalar reimplementation.
BOLTZ_TEST(mlp_block_matches_reference) {
    Ctx c;
    const int in = 3, hid = 4, out = 2;
    ggml_tensor* W1 = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, in, hid);
    ggml_tensor* b1 = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, hid);
    ggml_tensor* W2 = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, hid, out);
    ggml_tensor* b2 = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, out);
    ggml_tensor* x = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, in);

    std::vector<float> W1v = {0.1f, -0.2f, 0.3f,  0.0f, 0.5f, -0.1f,
                              0.2f, 0.2f,  -0.3f, 0.4f, -0.4f, 0.1f};
    std::vector<float> b1v = {0.01f, -0.02f, 0.03f, 0.0f};
    std::vector<float> W2v = {0.5f, -0.5f, 0.25f, 0.1f, -0.2f, 0.3f, 0.4f, -0.1f};
    std::vector<float> b2v = {0.05f, -0.05f};
    std::vector<float> xv = {1.0f, -0.5f, 2.0f};
    set(W1, W1v); set(b1, b1v); set(W2, W2v); set(b2, b2v); set(x, xv);

    ggml_tensor* h = boltz::nn::linear(c.ctx, x, W1, b1);
    h = boltz::nn::gelu(c.ctx, h);
    ggml_tensor* y = boltz::nn::linear(c.ctx, h, W2, b2);
    compute(c.ctx, y);

    // Scalar reference.
    std::vector<float> hr(hid);
    for (int o = 0; o < hid; ++o) {
        float s = b1v[o];
        for (int i = 0; i < in; ++i) s += W1v[o * in + i] * xv[i];
        hr[o] = gelu_ref(s);
    }
    std::vector<float> yr(out);
    for (int o = 0; o < out; ++o) {
        float s = b2v[o];
        for (int i = 0; i < hid; ++i) s += W2v[o * hid + i] * hr[i];
        yr[o] = s;
    }
    const float* yd = ggml_get_data_f32(y);
    for (int o = 0; o < out; ++o) expect_eq_f(yd[o], yr[o], "mlp out");
}

// BatchNorm1d eval-mode, folded to scale/shift and applied via affine, checked
// against the direct (x-mean)/sqrt(var+eps)*gamma+beta formula.
BOLTZ_TEST(batchnorm_fold_and_affine) {
    std::vector<float> mean = {0.5f, -1.0f, 2.0f};
    std::vector<float> var = {1.0f, 4.0f, 0.25f};
    std::vector<float> gamma = {2.0f, 1.0f, -1.0f};
    std::vector<float> beta = {0.1f, 0.2f, -0.3f};
    const float eps = 1e-5f;

    auto folded = boltz::nn::fold_batchnorm(mean, var, gamma, beta, eps);

    Ctx c;
    const int f = 3;
    ggml_tensor* x = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, f);
    ggml_tensor* scale = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, f);
    ggml_tensor* shift = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, f);
    std::vector<float> xv = {1.0f, 0.0f, 2.5f};
    set(x, xv);
    set(scale, folded.scale);
    set(shift, folded.shift);

    ggml_tensor* y = boltz::nn::affine(c.ctx, x, scale, shift);
    compute(c.ctx, y);

    const float* yd = ggml_get_data_f32(y);
    for (int i = 0; i < f; ++i) {
        float ref = (xv[i] - mean[i]) / std::sqrt(var[i] + eps) * gamma[i] + beta[i];
        expect_eq_f(yd[i], ref, "batchnorm eval");
    }
}

int main() {
    std::printf("== test_ggml_nn ==\n");
    return boltztest::run_all();
}
