#include "boltz/ggml_nn.hpp"

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

}  // namespace nn
}  // namespace boltz
