// Sequence-local atom attention windowing (boltzgen.model.modules.encoders
// get_indexing_matrix / single_to_keys, AF3 Algorithm 7). For each query window
// of W atoms, gathers a block of H keys from the surrounding half-windows
// (H = h * W/2, h = H/(W/2)), with out-of-range half-windows zero-filled. This
// is the key-gather that drives the windowed AtomTransformer.
#pragma once

#include <vector>

namespace boltz {

// single: [N*D] (N = K*W). Returns keys [K*H*D]: for query window k, the H keys
// gathered from half-windows j = 2k + (c+1) - h/2 for c in [0,h), each
// contributing W/2 atoms; missing half-windows are zero.
std::vector<float> single_to_keys(const std::vector<float>& single, int N, int D, int W,
                                  int H);

}  // namespace boltz
