// Hand-rolled GNN structural ops for the inverse-folding model: segment scatter
// (sum / max / softmax) of edge features into destination nodes, and the
// k-nearest-neighbour graph builder. Ports
// boltzgen.model.modules.scatter_utils and InverseFolding*.init_knn_graph
// (single-batch). Index/aggregation logic lives in C++; the learned MLPs that
// consume these edges run as ggml subgraphs.
#pragma once

#include <vector>

namespace boltz {

// src is [E, F] row-major; index[e] is the destination node of edge e.
// Returns [num_nodes, F]: out[n, f] = sum over edges e with index[e]==n of src[e, f].
std::vector<float> scatter_sum(const std::vector<float>& src, int E, int F,
                               const std::vector<int>& index, int num_nodes);

// Returns [num_nodes, F] with the per-group max; groups with no edge stay 0
// (mirrors scatter_max's include_self=False then -inf -> 0).
std::vector<float> scatter_max(const std::vector<float>& src, int E, int F,
                               const std::vector<int>& index, int num_nodes);

// Returns [E, F]: softmax of src within each destination group, per feature
// column (numerically stabilised by the group max; +1e-10 in the denominator,
// matching scatter_softmax).
std::vector<float> scatter_softmax(const std::vector<float>& src, int E, int F,
                                   const std::vector<int>& index, int num_nodes);

// k-NN graph over `num_nodes` 3-D points (coords is [N, 3] row-major).
// For each destination node we keep the `topk` nearest source nodes by
// Euclidean distance (self included, since self-distance is 0). Invalid nodes
// (valid[n]==false) are masked out (distance treated as +inf). Ties broken by
// ascending node index for determinism.
struct KnnGraph {
    std::vector<int> src_idx;  // length num_nodes * k
    std::vector<int> dst_idx;  // length num_nodes * k
    std::vector<char> edge_valid;  // valid[src] && valid[dst]
    int k = 0;
};

KnnGraph knn_graph(const std::vector<float>& coords, const std::vector<char>& valid,
                   int num_nodes, int topk);

}  // namespace boltz
