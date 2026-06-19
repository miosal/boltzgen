// Cross-language round-trip: the stdlib-only Python GGUF writer
// (tools/gen_sample_gguf.py) produces a file that the C++ WeightStore loads.
// This validates that the offline converter's output format is byte-compatible
// with ggml's loader. Skips (passes) if python3 is unavailable.
#include "boltz/weight_store.hpp"
#include "test_framework.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace boltz;
using namespace boltztest;

#ifndef BOLTZ_TOOLS_DIR
#define BOLTZ_TOOLS_DIR "."
#endif

static bool python_available() {
    return std::system("python3 --version >/dev/null 2>&1") == 0;
}

BOLTZ_TEST(python_writer_roundtrips_through_weight_store) {
    if (!python_available()) {
        std::printf("    (skip: python3 not available)\n");
        return;
    }
    const std::string out = "/tmp/boltzcpp_py_roundtrip.gguf";
    const std::string tools = BOLTZ_TOOLS_DIR;
    const std::string cmd =
        "cd '" + tools + "' && python3 gen_sample_gguf.py '" + out + "'";
    if (std::system(cmd.c_str()) != 0)
        throw AssertFailure("python gguf generator failed");

    WeightStore ws(out);
    expect_eq_i(ws.n_tensors(), 2, "two tensors");

    ggml_tensor* w = ws.get("layer.weight");
    expect_eq_i(w->ne[0], 3, "weight ne0 (in)");
    expect_eq_i(w->ne[1], 2, "weight ne1 (out)");
    const float* wd = ggml_get_data_f32(w);
    const float expect_w[6] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    for (int i = 0; i < 6; ++i) expect_eq_f(wd[i], expect_w[i], "weight val");

    const float* bd = ggml_get_data_f32(ws.get("layer.bias"));
    expect_eq_f(bd[0], -1.0f, "bias0");
    expect_eq_f(bd[1], 2.0f, "bias1");

    auto arch = ws.meta_str("general.architecture");
    expect_true(arch.has_value() && *arch == "boltzgen-ifold", "arch meta");
    auto nd = ws.meta_u32("boltzgen.node_dim");
    expect_true(nd.has_value() && *nd == 128u, "node_dim meta");

    std::remove(out.c_str());
}

int main() {
    std::printf("== test_python_gguf ==\n");
    return boltztest::run_all();
}
