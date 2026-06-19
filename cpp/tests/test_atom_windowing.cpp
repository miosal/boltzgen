// Test the sequence-local atom key-gather (single_to_keys) against a
// hand-derived windowing for K=2, W=4, H=8 (h=4, half-window=2 atoms).
#include "boltz/atom_windowing.hpp"
#include "test_framework.hpp"

#include <vector>

using namespace boltz;
using namespace boltztest;

BOLTZ_TEST(single_to_keys_windowing) {
    // N=8 atoms, D=1 so each atom value == its index.
    const int N = 8, D = 1, W = 4, H = 8;
    std::vector<float> single(N);
    for (int i = 0; i < N; ++i) single[i] = static_cast<float>(i);

    auto keys = single_to_keys(single, N, D, W, H);
    expect_eq_i(static_cast<long>(keys.size()), 2 * H * D, "K*H*D");

    // Derived: K=2, half-window=2, h=4, h/2=2.
    // Window 0: half-windows j = c+1-2 for c=0..3 => {-1,0,1,2}
    //   => [zeros, atoms0-1, atoms2-3, atoms4-5]
    const float w0[8] = {0, 0, 0, 1, 2, 3, 4, 5};
    for (int s = 0; s < 8; ++s) expect_eq_f(keys[0 * H + s], w0[s], "window0");

    // Window 1: half-windows j = 2 + c+1-2 = c+1 for c=0..3 => {1,2,3,4}
    //   => [atoms2-3, atoms4-5, atoms6-7, zeros]
    const float w1[8] = {2, 3, 4, 5, 6, 7, 0, 0};
    for (int s = 0; s < 8; ++s) expect_eq_f(keys[1 * H + s], w1[s], "window1");
}

BOLTZ_TEST(single_to_keys_multidim) {
    // D=2: confirm per-feature gather is correct on a small case.
    const int N = 4, D = 2, W = 4, H = 4;  // K=1, h=2, half-window=2
    std::vector<float> single(N * D);
    for (int i = 0; i < N * D; ++i) single[i] = static_cast<float>(i);
    auto keys = single_to_keys(single, N, D, W, H);
    // K=1: half-windows j = 2*0 + c+1 - 1 = c for c in {0,1} => hw0(atoms0,1), hw1(atoms2,3)
    // keys = [atom0, atom1, atom2, atom3] each D=2 => identical to input.
    expect_eq_i(static_cast<long>(keys.size()), H * D, "H*D");
    for (int i = 0; i < N * D; ++i) expect_eq_f(keys[i], single[i], "identity gather");
}

int main() {
    std::printf("== test_atom_windowing ==\n");
    return boltztest::run_all();
}
