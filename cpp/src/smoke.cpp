// Smoke test for the boltz.cpp scaffold.
//
// Purpose: prove the build constraints are satisfiable in this environment —
// ggml as a submodule, plain CMake, CPU backend — by building a tiny graph and
// running it. This is NOT a model; it just exercises ggml end to end so the
// toolchain is known-good before any Boltz kernels land.
//
// Computes y = A @ x where A is 2x3 and x is a 3-vector, then checks the result.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cmath>

int main() {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    // ggml_mul_mat(A, x): A has ne=[k, m], x has ne=[k], result has ne=[m].
    // Row i of the logical matrix is A->data[i*k .. i*k+k).
    const int k = 3, m = 2;
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);

    float * ad = ggml_get_data_f32(A);
    ad[0] = 1; ad[1] = 2; ad[2] = 3;   // row 0
    ad[3] = 4; ad[4] = 5; ad[5] = 6;   // row 1
    float * xd = ggml_get_data_f32(x);
    xd[0] = 1; xd[1] = 1; xd[2] = 1;

    struct ggml_tensor * y = ggml_mul_mat(ctx, A, x); // -> [m]

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/1) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    const float * yd = ggml_get_data_f32(y);
    const float expected[2] = {6.0f, 15.0f}; // 1+2+3, 4+5+6
    int rc = 0;
    for (int i = 0; i < m; ++i) {
        std::printf("y[%d] = %.1f (expected %.1f)\n", i, yd[i], expected[i]);
        if (std::fabs(yd[i] - expected[i]) > 1e-5f) rc = 1;
    }

    ggml_free(ctx);
    std::printf(rc == 0 ? "SMOKE OK: ggml CPU backend works.\n"
                        : "SMOKE FAIL\n");
    return rc;
}
