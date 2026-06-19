// Hand-rolled C++ featurization helpers (preprocessing, no ggml / no learned
// weights). Mirrors deterministic feature embeddings used by the BoltzGen
// models, e.g. GaussianSmearing for edge distances in inverse folding.
#pragma once

#include <vector>

namespace boltz {

// Row-major (rows x cols) float matrix.
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;
    float at(int r, int c) const { return data[r * cols + c]; }
    float& at(int r, int c) { return data[r * cols + c]; }
};

// Port of boltzgen.model.modules.inverse_fold.GaussianSmearing.
// offsets = linspace(start, stop, num_gaussians);
// coeff   = -0.5 / (offsets[1]-offsets[0])^2;
// out[i,k] = exp(coeff * (dist[i] - offsets[k])^2).
// Returns a (len(dist) x num_gaussians) matrix.
Matrix gaussian_smearing(const std::vector<float>& dist, float start, float stop,
                         int num_gaussians);

}  // namespace boltz
