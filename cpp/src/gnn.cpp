#include "boltz/gnn.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace boltz {

std::vector<float> scatter_sum(const std::vector<float>& src, int E, int F,
                               const std::vector<int>& index, int num_nodes) {
    std::vector<float> out(static_cast<size_t>(num_nodes) * F, 0.0f);
    for (int e = 0; e < E; ++e) {
        const int n = index[e];
        for (int f = 0; f < F; ++f) out[n * F + f] += src[e * F + f];
    }
    return out;
}

std::vector<float> scatter_max(const std::vector<float>& src, int E, int F,
                               const std::vector<int>& index, int num_nodes) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> out(static_cast<size_t>(num_nodes) * F, neg_inf);
    for (int e = 0; e < E; ++e) {
        const int n = index[e];
        for (int f = 0; f < F; ++f)
            out[n * F + f] = std::max(out[n * F + f], src[e * F + f]);
    }
    // Untouched groups (still -inf) become 0, matching the Python helper.
    for (float& v : out)
        if (std::isinf(v)) v = 0.0f;
    return out;
}

std::vector<float> scatter_softmax(const std::vector<float>& src, int E, int F,
                                   const std::vector<int>& index, int num_nodes) {
    const std::vector<float> mx = scatter_max(src, E, F, index, num_nodes);
    std::vector<float> exp_src(static_cast<size_t>(E) * F);
    for (int e = 0; e < E; ++e) {
        const int n = index[e];
        for (int f = 0; f < F; ++f)
            exp_src[e * F + f] = std::exp(src[e * F + f] - mx[n * F + f]);
    }
    const std::vector<float> sum_exp = scatter_sum(exp_src, E, F, index, num_nodes);
    std::vector<float> out(static_cast<size_t>(E) * F);
    for (int e = 0; e < E; ++e) {
        const int n = index[e];
        for (int f = 0; f < F; ++f)
            out[e * F + f] = exp_src[e * F + f] / (sum_exp[n * F + f] + 1e-10f);
    }
    return out;
}

KnnGraph knn_graph(const std::vector<float>& coords, const std::vector<char>& valid,
                   int num_nodes, int topk) {
    const int k = std::min(topk, num_nodes);
    const float inf = std::numeric_limits<float>::infinity();
    KnnGraph g;
    g.k = k;
    g.src_idx.reserve(static_cast<size_t>(num_nodes) * k);
    g.dst_idx.reserve(static_cast<size_t>(num_nodes) * k);
    g.edge_valid.reserve(static_cast<size_t>(num_nodes) * k);

    auto dist = [&](int i, int j) -> float {
        if (!valid[i] || !valid[j]) return inf;  // masked pair
        const float dx = coords[i * 3 + 0] - coords[j * 3 + 0];
        const float dy = coords[i * 3 + 1] - coords[j * 3 + 1];
        const float dz = coords[i * 3 + 2] - coords[j * 3 + 2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    std::vector<int> order(num_nodes);
    for (int dst = 0; dst < num_nodes; ++dst) {
        for (int j = 0; j < num_nodes; ++j) order[j] = j;
        // Smallest distance first; ties broken by ascending index (deterministic).
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const float da = dist(dst, a), db = dist(dst, b);
            if (da != db) return da < db;
            return a < b;
        });
        for (int t = 0; t < k; ++t) {
            const int src = order[t];
            g.src_idx.push_back(src);
            g.dst_idx.push_back(dst);
            g.edge_valid.push_back((valid[src] && valid[dst]) ? 1 : 0);
        }
    }
    return g;
}

}  // namespace boltz
