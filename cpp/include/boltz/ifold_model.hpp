// Inverse-folding inference model assembled from validated components:
//   residues -> CA coords -> KNN graph -> token+edge embedding ->
//   N x MLPAttnGNN encoder (ggml) -> LayerNorm+Linear head -> per-residue tokens.
//
// This is the structural GNN core of boltzgen's inverse folder, wired end to end
// so the pipeline RUNS on a real structure. The exact input embedder and the
// autoregressive decoder of the reference model are simplified here (single
// forward, argmax/greedy) and weights come from a self-describing gguf; swapping
// in the converted checkpoint is the remaining integration step.
#pragma once

#include "boltz/cif.hpp"
#include "boltz/gnn_layer.hpp"
#include "boltz/weight_store.hpp"

#include <string>
#include <vector>

namespace boltz {

struct IFoldConfig {
    int node_dim = 0;
    int pair_dim = 0;
    int hidden_dim = 0;
    int num_heads = 0;
    int num_layers = 0;
    int num_gaussians = 0;
    int topk = 0;
    int num_embed = 0;   // token embedding rows (20 canonical + mask)
    float g_start = 0.0f;
    float g_stop = 20.0f;
};

struct IFoldWeights {
    std::vector<float> tok_embed;          // num_embed * node_dim
    Linear edge_proj;                      // num_gaussians -> pair_dim
    std::vector<MLPAttnGNNWeights> layers; // num_layers
    std::vector<float> head_ln_w;          // node_dim
    std::vector<float> head_ln_b;          // node_dim
    Linear head;                           // node_dim -> 20
};

struct IFoldModel {
    IFoldConfig cfg;
    IFoldWeights w;
};

// Load config (from gguf metadata) + weights (by name) from a converted/synthetic
// inverse-folding gguf.
IFoldModel load_ifold_model(const WeightStore& ws);

struct IFoldResult {
    std::vector<int> tokens;   // canonical index [0,20) per residue
    std::string sequence;      // one-letter sequence
};

// Run inference. `design_mask[i]` true => residue i is being designed (its input
// identity is masked); false => its known residue type conditions the prediction.
IFoldResult run_ifold(const IFoldModel& m, const std::vector<Residue>& residues,
                      const std::vector<char>& design_mask);

}  // namespace boltz
