// MLPAttnGNN encoder layer (boltzgen.model.modules.inverse_fold.MLPAttnGNN),
// assembled from the tested primitives: dense MLPs run as ggml matmul+gelu
// graphs, while gather / concat / scatter / batchnorm-affine run in C++.
// Inference only (dropout is a no-op; BatchNorm uses folded running stats).
//
// Shapes (assuming hidden_dim == node_dim, as in the shipped config):
//   s : [N, node]   node features
//   z : [E, pair]   edge features
//   src[e], dst[e]  endpoints of edge e
#pragma once

#include <vector>

namespace boltz {

// A Linear layer: weight is [out, in] row-major (PyTorch layout), bias is [out].
struct Linear {
    std::vector<float> w;  // out*in
    std::vector<float> b;  // out
    int in = 0;
    int out = 0;
};

// Folded BatchNorm1d (eval): per-feature scale/shift (see nn::fold_batchnorm).
struct Affine {
    std::vector<float> scale;  // F
    std::vector<float> shift;  // F
};

struct MLPAttnGNNWeights {
    int node_dim = 0;
    int pair_dim = 0;
    int hidden_dim = 0;
    int num_heads = 0;

    // edge_FFN: Linear -> GELU -> Linear -> BatchNorm
    Linear edge_l1, edge_l2;
    Affine edge_bn;

    // attn_weight_mlp: Linear -> GELU -> Linear -> GELU -> Linear(num_heads)
    Linear aw_l1, aw_l2, aw_l3;

    // attn_value_mlp: Linear -> GELU -> Linear -> GELU -> Linear(node_dim)
    Linear av_l1, av_l2, av_l3;

    // attn_output_linear: Linear(hidden*num_heads -> node) -> BatchNorm
    Linear out_l;
    Affine out_bn;

    // attn_FFN: Linear -> GELU -> Linear -> BatchNorm
    Linear ffn_l1, ffn_l2;
    Affine ffn_bn;
};

struct MLPAttnGNNState {
    std::vector<float> s;  // [N, node]
    std::vector<float> z;  // [E, pair]
};

// Runs one MLPAttnGNN forward. `s` is [N*node], `z` is [E*pair].
MLPAttnGNNState mlp_attn_gnn_forward(const MLPAttnGNNWeights& w, int num_nodes,
                                     const std::vector<float>& s,
                                     const std::vector<float>& z,
                                     const std::vector<int>& src,
                                     const std::vector<int>& dst);

// Apply a single Linear to `rows` input vectors (x is [rows*L.in] row-major).
// Returns [rows*L.out]. Runs as one ggml matmul+bias graph, matching the MLP path.
std::vector<float> linear_batch(const Linear& L, const std::vector<float>& x, int rows);

// One MLPAttnGNNDecoder forward (boltzgen.model.modules.inverse_fold
// .MLPAttnGNNDecoder). Unlike the encoder layer there is no edge_FFN and the
// edge tensor `nbr` (the decoder's "neighbors_rep", width `neighbor_dim`) is not
// updated. `s` is [num_nodes*node], `nbr` is [E*neighbor_dim]. Returns the
// updated [num_nodes*node] node features.
std::vector<float> mlp_attn_gnn_decoder_forward(const MLPAttnGNNWeights& w, int num_nodes,
                                                const std::vector<float>& s,
                                                const std::vector<float>& nbr,
                                                const std::vector<int>& src,
                                                const std::vector<int>& dst,
                                                int neighbor_dim);

}  // namespace boltz
