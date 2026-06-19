// Tests the golden-tensor parity harness: the C++ .npy reader + compare().
// Cross-language round-trip (stdlib Python writer -> C++ reader) proves the
// fixture format is compatible, so real PyTorch-dumped reference tensors will
// load here for numeric parity. Skips if python3 is unavailable.
#include "boltz/npy.hpp"
#include "test_framework.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace boltz;
using namespace boltztest;

#ifndef BOLTZ_TOOLS_DIR
#define BOLTZ_TOOLS_DIR "."
#endif

static bool python_available() { return std::system("python3 --version >/dev/null 2>&1") == 0; }

BOLTZ_TEST(npy_roundtrip_and_compare) {
    if (!python_available()) { std::printf("    (skip: no python3)\n"); return; }
    const std::string out = "/tmp/boltzcpp_test.npy";
    const std::string tools = BOLTZ_TOOLS_DIR;
    const std::string cmd = "cd '" + tools + "' && python3 npy_writer.py '" + out + "'";
    if (std::system(cmd.c_str()) != 0) throw AssertFailure("npy writer failed");

    NpyArray a = load_npy(out);
    expect_eq_i(static_cast<long>(a.shape.size()), 2, "2-D");
    expect_eq_i(a.shape[0], 2, "rows");
    expect_eq_i(a.shape[1], 3, "cols");
    const float expect[6] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; ++i) expect_eq_f(a.data[i], expect[i], "npy value");

    // compare(): identical arrays -> zero diff; perturbed -> reflects max/mean.
    DiffStats same = compare(a, a);
    expect_true(same.shapes_match, "shapes match");
    expect_eq_f(same.max_abs, 0.0f, "identical max 0");

    NpyArray b = a;
    b.data[0] += 0.5f;
    b.data[5] -= 0.25f;
    DiffStats d = compare(a, b);
    expect_eq_f(d.max_abs, 0.5f, "max abs diff");
    expect_true(d.mean_abs > 0.0f, "nonzero mean");

    std::remove(out.c_str());
}

BOLTZ_TEST(npy_missing_file_throws) {
    expect_throws_contains([] { load_npy("/tmp/boltzcpp_no_such.npy"); }, "cannot open", "missing");
}

int main() {
    std::printf("== test_npy ==\n");
    return boltztest::run_all();
}
