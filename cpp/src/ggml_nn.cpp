#include "boltz/ggml_nn.hpp"

#include <cmath>

namespace boltz {
namespace nn {

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);  // [out, ...]
    if (b != nullptr) y = ggml_add(ctx, y, b);
    return y;
}

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                        ggml_tensor* b, float eps) {
    ggml_tensor* y = ggml_norm(ctx, x, eps);  // (x - mean) / sqrt(var + eps) over ne0
    if (w != nullptr) y = ggml_mul(ctx, y, w);
    if (b != nullptr) y = ggml_add(ctx, y, b);
    return y;
}

ggml_tensor* gelu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu_erf(ctx, x);
}

ggml_tensor* affine(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale,
                    ggml_tensor* shift) {
    return ggml_add(ctx, ggml_mul(ctx, x, scale), shift);
}

FoldedAffine fold_batchnorm(const std::vector<float>& mean, const std::vector<float>& var,
                            const std::vector<float>& gamma, const std::vector<float>& beta,
                            float eps) {
    const size_t f = mean.size();
    FoldedAffine out;
    out.scale.resize(f);
    out.shift.resize(f);
    for (size_t i = 0; i < f; ++i) {
        const float s = gamma[i] / std::sqrt(var[i] + eps);
        out.scale[i] = s;
        out.shift[i] = beta[i] - mean[i] * s;
    }
    return out;
}

}  // namespace nn
}  // namespace boltz
