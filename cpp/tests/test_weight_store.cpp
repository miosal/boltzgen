// Round-trip test for the gguf weight store: write a small gguf with named
// tensors + metadata using ggml's gguf API, then load it through WeightStore
// and verify names, shapes, values and metadata survive. No model weights
// required — this validates the weight-loading plumbing the converter targets.
#include "boltz/weight_store.hpp"
#include "test_framework.hpp"

#include "gguf.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace boltz;
using namespace boltztest;

static std::string write_sample_gguf() {
    std::string path = "/tmp/boltzcpp_test_weights.gguf";

    ggml_init_params ip{16 * 1024 * 1024, nullptr, false};
    ggml_context* ctx = ggml_init(ip);

    // A [in=3, out=2] linear weight and its bias.
    ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_set_name(w, "layer.weight");
    float* wd = ggml_get_data_f32(w);
    for (int i = 0; i < 6; ++i) wd[i] = static_cast<float>(i) + 0.5f;

    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_set_name(b, "layer.bias");
    float* bd = ggml_get_data_f32(b);
    bd[0] = -1.0f;
    bd[1] = 2.0f;

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "boltzgen-ifold");
    gguf_set_val_u32(g, "boltzgen.node_dim", 128u);
    gguf_add_tensor(g, w);
    gguf_add_tensor(g, b);
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);

    gguf_free(g);
    ggml_free(ctx);
    return path;
}

BOLTZ_TEST(weight_store_roundtrip) {
    const std::string path = write_sample_gguf();
    WeightStore ws(path);

    expect_eq_i(ws.n_tensors(), 2, "two tensors");
    expect_true(ws.has("layer.weight"), "weight present");
    expect_true(ws.has("layer.bias"), "bias present");
    expect_true(!ws.has("does.not.exist"), "absent reported");

    ggml_tensor* w = ws.get("layer.weight");
    expect_eq_i(w->ne[0], 3, "weight ne0");
    expect_eq_i(w->ne[1], 2, "weight ne1");
    const float* wd = ggml_get_data_f32(w);
    for (int i = 0; i < 6; ++i) expect_eq_f(wd[i], static_cast<float>(i) + 0.5f, "weight val");

    const float* bd = ggml_get_data_f32(ws.get("layer.bias"));
    expect_eq_f(bd[0], -1.0f, "bias0");
    expect_eq_f(bd[1], 2.0f, "bias1");

    auto arch = ws.meta_str("general.architecture");
    expect_true(arch.has_value() && *arch == "boltzgen-ifold", "arch meta");
    auto nd = ws.meta_u32("boltzgen.node_dim");
    expect_true(nd.has_value() && *nd == 128u, "node_dim meta");
    expect_true(!ws.meta_str("missing.key").has_value(), "absent meta");

    std::remove(path.c_str());
}

BOLTZ_TEST(weight_store_missing_file_throws) {
    expect_throws_contains([] { WeightStore ws("/tmp/boltzcpp_no_such_file.gguf"); },
                           "failed to load", "missing file");
}

int main() {
    std::printf("== test_weight_store ==\n");
    return boltztest::run_all();
}
