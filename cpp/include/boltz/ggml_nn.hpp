// Thin neural-net building blocks expressed as ggml subgraphs. These are the
// primitives the BoltzGen models are assembled from (Linear, LayerNorm, GELU,
// SiLU, sigmoid). Building them as ggml ops — rather than scalar C++ — is what
// lets the same graph run on the CPU backend now and the Vulkan backend later.
#pragma once

#include "ggml.h"

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

}  // namespace nn
}  // namespace boltz
