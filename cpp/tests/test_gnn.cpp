// Tests for GNN scatter ops + KNN graph. Expected values hand-computed from the
// semantics of boltzgen.model.modules.scatter_utils and init_knn_graph.
#include "boltz/gnn.hpp"
#include "test_framework.hpp"

#include <cmath>

using namespace boltz;
using namespace boltztest;

// 3 edges into 2 nodes, 2 features.
//   edge0 -> node0 : [1, 2]
//   edge1 -> node1 : [3, 4]
//   edge2 -> node0 : [5, 6]
static const std::vector<float> SRC = {1, 2, 3, 4, 5, 6};
static const std::vector<int> IDX = {0, 1, 0};

BOLTZ_TEST(scatter_sum_basic) {
    auto out = scatter_sum(SRC, /*E=*/3, /*F=*/2, IDX, /*N=*/2);
    expect_eq_f(out[0], 6.0f, "node0 f0 = 1+5");
    expect_eq_f(out[1], 8.0f, "node0 f1 = 2+6");
    expect_eq_f(out[2], 3.0f, "node1 f0");
    expect_eq_f(out[3], 4.0f, "node1 f1");
}

BOLTZ_TEST(scatter_max_basic_and_empty) {
    auto out = scatter_max(SRC, 3, 2, IDX, /*N=*/3);
    expect_eq_f(out[0], 5.0f, "node0 f0 max(1,5)");
    expect_eq_f(out[1], 6.0f, "node0 f1 max(2,6)");
    expect_eq_f(out[2], 3.0f, "node1 f0");
    // node2 has no incoming edge -> 0
    expect_eq_f(out[4], 0.0f, "empty group -> 0");
    expect_eq_f(out[5], 0.0f, "empty group -> 0");
}

BOLTZ_TEST(scatter_softmax_normalizes_per_group) {
    auto out = scatter_softmax(SRC, 3, 2, IDX, /*N=*/2);
    // Node0 group = edges 0 and 2; per feature the two outputs must sum to ~1.
    expect_eq_f(out[0 * 2 + 0] + out[2 * 2 + 0], 1.0f, "group0 f0 sums to 1");
    expect_eq_f(out[0 * 2 + 1] + out[2 * 2 + 1], 1.0f, "group0 f1 sums to 1");
    // Node1 group = single edge -> softmax is ~1.
    expect_true(std::fabs(out[1 * 2 + 0] - 1.0f) < 1e-6f, "singleton -> 1");
    // Larger logit gets larger weight (edge2 f0=5 > edge0 f0=1).
    expect_true(out[2 * 2 + 0] > out[0 * 2 + 0], "monotone in logit");
}

BOLTZ_TEST(knn_graph_self_and_counts) {
    // 4 collinear points; topk=2.
    std::vector<float> coords = {0, 0, 0,  1, 0, 0,  2, 0, 0,  3, 0, 0};
    std::vector<char> valid = {1, 1, 1, 1};
    auto g = knn_graph(coords, valid, /*N=*/4, /*topk=*/2);
    expect_eq_i(g.k, 2, "k");
    expect_eq_i(static_cast<long>(g.src_idx.size()), 8, "N*k edges");
    // For each destination, the first (nearest) neighbour is itself (dist 0).
    for (int dst = 0; dst < 4; ++dst) {
        const int first_src = g.src_idx[dst * g.k + 0];
        expect_eq_i(first_src, dst, "self is nearest");
        expect_eq_i(g.dst_idx[dst * g.k + 0], dst, "dst label");
    }
    // Node 0's second neighbour is node 1 (closest non-self).
    expect_eq_i(g.src_idx[0 * g.k + 1], 1, "node0 nn = node1");
}

BOLTZ_TEST(knn_graph_invalid_nodes_masked) {
    std::vector<float> coords = {0, 0, 0,  1, 0, 0,  2, 0, 0};
    std::vector<char> valid = {1, 0, 1};  // node1 invalid
    auto g = knn_graph(coords, valid, 3, /*topk=*/3);
    // Edges into an invalid destination, or from an invalid source, are flagged.
    for (size_t e = 0; e < g.src_idx.size(); ++e) {
        const bool expect_valid = valid[g.src_idx[e]] && valid[g.dst_idx[e]];
        expect_true((g.edge_valid[e] != 0) == expect_valid, "edge_valid flag");
    }
}

int main() {
    std::printf("== test_gnn ==\n");
    return boltztest::run_all();
}
