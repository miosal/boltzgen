// Minimal hand-rolled .npy reader (NumPy format v1.0) + array comparison, for
// the golden-tensor parity harness. PyTorch-side fixtures (reference inputs and
// outputs dumped as .npy) are loaded here and compared against the C++ pipeline
// output, so numeric parity becomes a `ctest` once the fixtures exist. Stdlib
// only; supports C-order float32 / float64.
#pragma once

#include <string>
#include <vector>

namespace boltz {

struct NpyArray {
    std::vector<long> shape;
    std::vector<float> data;  // always materialized as float
    long size() const {
        long n = 1;
        for (long d : shape) n *= d;
        return n;
    }
};

// Load a .npy file (float32/float64, C-order). Throws std::runtime_error on
// malformed input or unsupported dtype/order.
NpyArray load_npy(const std::string& path);

// Comparison result for parity checks.
struct DiffStats {
    long n = 0;
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    bool shapes_match = true;
};

DiffStats compare(const NpyArray& got, const NpyArray& golden);

}  // namespace boltz
