// Hand-rolled C++ featurization helpers (preprocessing, no ggml / no learned
// weights). Mirrors deterministic feature embeddings used by the BoltzGen
// models, e.g. GaussianSmearing for edge distances in inverse folding.
#pragma once

#include <string>
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

// Token-level features (the portable, non-chemistry part of
// boltzgen.data.feature.featurizer.process_token_features). Built from a
// chain's residues.
struct TokenFeatures {
    int n = 0;
    int num_token_types = 0;          // 33
    std::vector<float> res_type;       // [n * num_token_types] one-hot
    std::vector<int> token_index;      // [n] 0..n-1
    std::vector<int> residue_index;    // [n] (from seq_id)
    std::vector<int> mol_type;         // [n]
    std::vector<float> pad_mask;       // [n] (1 = present)
};

// Build token features from residue names + indices. comp_ids[i] is the residue
// 3-letter name; seq_ids[i] its (CIF label) index.
TokenFeatures token_features(const std::vector<std::string>& comp_ids,
                             const std::vector<int>& seq_ids);

// Port of boltzgen.model.modules.inverse_fold.GaussianSmearing.
// offsets = linspace(start, stop, num_gaussians);
// coeff   = -0.5 / (offsets[1]-offsets[0])^2;
// out[i,k] = exp(coeff * (dist[i] - offsets[k])^2).
// Returns a (len(dist) x num_gaussians) matrix.
Matrix gaussian_smearing(const std::vector<float>& dist, float start, float stop,
                         int num_gaussians);

}  // namespace boltz
