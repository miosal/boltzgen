#include "boltz/atom_windowing.hpp"

#include <stdexcept>

namespace boltz {

std::vector<float> single_to_keys(const std::vector<float>& single, int N, int D, int W,
                                  int H) {
    if (W % 2 != 0) throw std::invalid_argument("single_to_keys: W must be even");
    const int hw = W / 2;          // half-window size
    if (H % hw != 0) throw std::invalid_argument("single_to_keys: H must be a multiple of W/2");
    const int h = H / hw;          // number of half-windows per key block
    const int K = N / W;           // number of query windows
    const int two_k = 2 * K;       // number of half-windows total

    std::vector<float> keys((size_t)K * H * D, 0.0f);
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < h; ++c) {
            const int j = 2 * k + (c + 1) - h / 2;  // source half-window
            if (j < 0 || j >= two_k) continue;       // out of range -> zeros
            for (int i = 0; i < hw; ++i) {
                const int src_atom = j * hw + i;            // original atom index
                const int key_slot = c * hw + i;            // slot within the H block
                for (int d = 0; d < D; ++d)
                    keys[((size_t)k * H + key_slot) * D + d] = single[(size_t)src_atom * D + d];
            }
        }
    }
    return keys;
}

}  // namespace boltz
