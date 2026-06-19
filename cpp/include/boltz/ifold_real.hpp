// Real-weights inverse-folding encoder + decoder, loaded from a converted
// BoltzGen checkpoint (boltzgen1_ifold.ckpt -> .gguf). Unlike ifold_model.hpp
// (a simplified standalone demo on synthetic weights), this matches the
// reference architecture exactly so it reproduces the PyTorch model's numerics:
//   * inverse_folding_encoder: linear_token_to_node, linear_token_to_pair,
//     N x MLPAttnGNN  (boltzgen.model.modules.inverse_fold.InverseFoldingEncoder)
//   * structure_module:        seq_to_s, M x MLPAttnGNNDecoder, predictor
//     (boltzgen.model.modules.inverse_fold.InverseFoldingDecoder.forward)
// BatchNorm (SyncBatchNorm, eval) is folded into per-feature scale/shift.
//
// The node features `s_inputs` (from the upstream InputEmbedder) and the edge
// features are supplied by the caller — the parity harness feeds tensors dumped
// from the reference (tools/dump_golden.py) and checks the outputs match.
#pragma once

#include "boltz/gnn_layer.hpp"
#include "boltz/weight_store.hpp"

#include <vector>

namespace boltz {

struct RealIFoldConfig {
    int node_dim = 0;     // 128
    int pair_dim = 0;     // 128
    int token_s = 0;      // 384 (s_inputs width = linear_token_to_node input)
    int edge_in = 0;      // 332 (linear_token_to_pair input)
    int num_heads = 0;    // 4
    int n_enc = 0;        // encoder MLPAttnGNN layers
    int n_dec = 0;        // decoder MLPAttnGNNDecoder layers
    int num_tokens = 0;   // 33 (predictor output / seq_to_s input)
    int neighbor_dim = 0; // pair_dim + node_dim (decoder edge width)
};

struct RealIFoldModel {
    RealIFoldConfig cfg;
    Linear linear_token_to_node;             // token_s -> node
    Linear linear_token_to_pair;             // edge_in -> pair
    std::vector<MLPAttnGNNWeights> enc;       // encoder layers
    std::vector<MLPAttnGNNWeights> dec;       // decoder layers (edge_* unused)
    Linear seq_to_s;                          // num_tokens -> node
    Linear predictor;                         // node -> num_tokens (no bias)
};

// Load from a converted inverse-fold gguf (verbatim PyTorch tensor names).
RealIFoldModel load_real_ifold_model(const WeightStore& ws);

struct RealEncoderOut {
    std::vector<float> s;  // [N * node_dim]
    std::vector<float> z;  // [E * pair_dim]
};

// Encoder: s = linear_token_to_node(s_inputs); z = linear_token_to_pair(pair_input);
// then the MLPAttnGNN stack. `s_inputs` is [N*token_s], `pair_input` is [E*edge_in].
RealEncoderOut run_real_encoder(const RealIFoldModel& m,
                                const std::vector<float>& s_inputs,
                                const std::vector<float>& pair_input,
                                const std::vector<int>& src,
                                const std::vector<int>& dst, int N);

// Decoder forward logits. `s_enc`/`z_enc` are the encoder outputs; `res_type_vis`
// is the per-edge (post-RNG) sequence-visibility one-hot fed to seq_to_s
// ([E*num_tokens]). Returns per-node logits [N*num_tokens].
std::vector<float> run_real_decoder_logits(const RealIFoldModel& m,
                                           const std::vector<float>& s_enc,
                                           const std::vector<float>& z_enc,
                                           const std::vector<float>& res_type_vis,
                                           const std::vector<int>& src,
                                           const std::vector<int>& dst, int N);

}  // namespace boltz
