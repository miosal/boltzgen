// Thin neural-net building blocks expressed as ggml subgraphs. These are the
// primitives the BoltzGen models are assembled from (Linear, LayerNorm, GELU,
// SiLU, sigmoid). Building them as ggml ops — rather than scalar C++ — is what
// lets the same graph run on the CPU backend now and the Vulkan backend later.
#pragma once

#include "ggml.h"

#include <vector>

namespace boltz {
namespace nn {

// y = W x (+ b).  Weight layout follows PyTorch nn.Linear: logical shape
// [out, in], stored as a ggml tensor with ne = [in, out]. `b` may be null.
ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b);

// LayerNorm over ne0 (the feature dim), matching torch.nn.LayerNorm with
// elementwise affine (population variance, like PyTorch unbiased=False).
// `w` (gamma) and `b` (beta) have ne = [features]; either may be null.
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                        ggml_tensor* b, float eps = 1e-5f);

// Exact (erf) GELU — matches torch.nn.GELU() default.
ggml_tensor* gelu(ggml_context* ctx, ggml_tensor* x);

// SiLU / swish — matches torch.nn.SiLU().
ggml_tensor* silu(ggml_context* ctx, ggml_tensor* x);

// SwiGLU Transition block (boltzgen.model.layers.transition.Transition):
//   x = norm(x); out = fc3( silu(fc1(x)) * fc2(x) ).  All Linears are bias-free.
// fc1/fc2 have ne=[dim, hidden]; fc3 has ne=[hidden, out].
ggml_tensor* transition(ggml_context* ctx, ggml_tensor* x, ggml_tensor* norm_w,
                        ggml_tensor* norm_b, ggml_tensor* fc1, ggml_tensor* fc2,
                        ggml_tensor* fc3, float eps = 1e-5f);

// y = x * scale + shift, broadcast over ne0 (scale/shift have ne = [features]).
ggml_tensor* affine(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale,
                    ggml_tensor* shift);

// Fold BatchNorm1d eval-mode params (running_mean/var + affine weight/bias) into
// a single per-feature scale/shift so inference is one `affine` op — the
// standard approach for a backend-portable graph. Mirrors:
//   y = (x - mean)/sqrt(var+eps) * gamma + beta
//     = x * (gamma/sqrt(var+eps)) + (beta - mean*gamma/sqrt(var+eps)).
struct FoldedAffine {
    std::vector<float> scale;
    std::vector<float> shift;
};
FoldedAffine fold_batchnorm(const std::vector<float>& mean, const std::vector<float>& var,
                            const std::vector<float>& gamma, const std::vector<float>& beta,
                            float eps = 1e-5f);

}  // namespace nn
}  // namespace boltz
