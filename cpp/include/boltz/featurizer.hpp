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

// Single-sequence ("no MSA") features — the inference path when no MSA is
// supplied (common for design). The MSA is just the query sequence: one row, a
// one-hot profile, and zero deletions. Mirrors the dummy-MSA featurization.
struct MsaFeatures {
    int n = 0;
    int num_token_types = 0;
    int depth = 1;                  // one row (the query)
    std::vector<float> msa;          // [depth*n*num_token_types] one-hot
    std::vector<float> profile;      // [n*num_token_types] (column token frequencies)
    std::vector<float> deletion_mean; // [n] per-column mean deletion
    std::vector<float> has_deletion; // [depth*n] = 0
    std::vector<float> deletion_value; // [depth*n] = 0
    std::vector<float> msa_mask;     // [depth*n] = 1
};

// `res_type` is the [n*num_token_types] one-hot from token_features.
MsaFeatures single_sequence_msa_features(const std::vector<float>& res_type, int n,
                                         int num_token_types);

// Multi-sequence MSA featurization from an explicit alignment (the portable
// per-alignment logic; obtaining the alignment from MSA database files +
// taxonomy pairing is a separate, data-gated step). `seq_tokens[d][c]` is the
// token id of sequence d at column c; `deletions[d][c]` the deletion count.
// Produces the one-hot MSA, the column profile (token frequencies over the
// alignment), and per-column mean deletion.
MsaFeatures msa_features_from_alignment(const std::vector<std::vector<int>>& seq_tokens,
                                        const std::vector<std::vector<float>>& deletions,
                                        int n, int num_token_types);

// Port of boltzgen.model.modules.inverse_fold.GaussianSmearing.
// offsets = linspace(start, stop, num_gaussians);
// coeff   = -0.5 / (offsets[1]-offsets[0])^2;
// out[i,k] = exp(coeff * (dist[i] - offsets[k])^2).
// Returns a (len(dist) x num_gaussians) matrix.
Matrix gaussian_smearing(const std::vector<float>& dist, float start, float stop,
                         int num_gaussians);

}  // namespace boltz
