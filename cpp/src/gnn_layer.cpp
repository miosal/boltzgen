#include "boltz/gnn_layer.hpp"

#include "boltz/gnn.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <stdexcept>

namespace boltz {
namespace {

struct DenseLayer {
    const Linear* lin;
    bool gelu;
};

// Runs a stack of Linear(+optional GELU) layers over a batch of `rows` vectors
// as a single ggml graph on the CPU backend. `input` is [rows*in0] row-major
// (row-major == ggml ne=[in0, rows]). Returns [rows*out_last] row-major.
std::vector<float> dense_stack(const std::vector<float>& input, int rows, int in0,
                               const std::vector<DenseLayer>& layers) {
    ggml_init_params ip{(size_t)64 * 1024 * 1024, nullptr, false};
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in0, rows);
    {
        float* xd = ggml_get_data_f32(x);
        for (size_t i = 0; i < input.size(); ++i) xd[i] = input[i];
    }

    int cur_in = in0;
    for (const DenseLayer& dl : layers) {
        const Linear& L = *dl.lin;
        if (L.in != cur_in)
            throw std::runtime_error("dense_stack: dim mismatch");
        ggml_tensor* W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L.in, L.out);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, L.out);
        {
            float* wd = ggml_get_data_f32(W);
            for (size_t i = 0; i < L.w.size(); ++i) wd[i] = L.w[i];
            float* bd = ggml_get_data_f32(b);
            for (size_t i = 0; i < L.b.size(); ++i) bd[i] = L.b[i];
        }
        x = ggml_add(ctx, ggml_mul_mat(ctx, W, x), b);
        if (dl.gelu) x = ggml_gelu_erf(ctx, x);
        cur_in = L.out;
    }

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, x);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("dense_stack: compute failed");
    }

    const float* yd = ggml_get_data_f32(x);
    std::vector<float> out((size_t)rows * cur_in);
    for (size_t i = 0; i < out.size(); ++i) out[i] = yd[i];
    ggml_free(ctx);
    return out;
}

// Concatenate per-row feature blocks (each blocks[k] is [rows*dim_k]) into
// [rows*sum(dim)] row-major.
std::vector<float> concat_rows(int rows,
                               const std::vector<std::pair<const std::vector<float>*, int>>& blocks) {
    int total = 0;
    for (auto& bl : blocks) total += bl.second;
    std::vector<float> out((size_t)rows * total);
    for (int r = 0; r < rows; ++r) {
        int off = 0;
        for (auto& bl : blocks) {
            const std::vector<float>& src = *bl.first;
            const int d = bl.second;
            for (int c = 0; c < d; ++c) out[r * total + off + c] = src[r * d + c];
            off += d;
        }
    }
    return out;
}

// Gather rows of `s` ([N*dim]) by `idx` (length E) -> [E*dim].
std::vector<float> gather(const std::vector<float>& s, int dim,
                          const std::vector<int>& idx) {
    std::vector<float> out((size_t)idx.size() * dim);
    for (size_t e = 0; e < idx.size(); ++e)
        for (int c = 0; c < dim; ++c) out[e * dim + c] = s[idx[e] * dim + c];
    return out;
}

void apply_affine(std::vector<float>& x, int rows, int F, const Affine& a) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < F; ++c) x[r * F + c] = x[r * F + c] * a.scale[c] + a.shift[c];
}

}  // namespace

MLPAttnGNNState mlp_attn_gnn_forward(const MLPAttnGNNWeights& w, int num_nodes,
                                     const std::vector<float>& s_in,
                                     const std::vector<float>& z_in,
                                     const std::vector<int>& src,
                                     const std::vector<int>& dst) {
    const int node = w.node_dim, pair = w.pair_dim, heads = w.num_heads;
    const int E = static_cast<int>(src.size());

    std::vector<float> s = s_in;  // [N, node]
    std::vector<float> z = z_in;  // [E, pair]

    const std::vector<float> s_src = gather(s, node, src);  // [E, node]
    const std::vector<float> s_dst = gather(s, node, dst);  // [E, node]

    // --- edge update: z = z + BN(edge_FFN(cat[s_src, s_dst, z])) ---
    {
        auto in = concat_rows(E, {{&s_src, node}, {&s_dst, node}, {&z, pair}});
        auto upd = dense_stack(in, E, 2 * node + pair,
                               {{&w.edge_l1, true}, {&w.edge_l2, false}});  // [E, pair]
        apply_affine(upd, E, pair, w.edge_bn);
        for (int i = 0; i < E * pair; ++i) z[i] += upd[i];
    }

    // --- attention weights: [E, heads] ---
    auto aw_in = concat_rows(E, {{&s_dst, node}, {&s_src, node}, {&z, pair}});
    auto attn_weight = dense_stack(aw_in, E, 2 * node + pair,
                                   {{&w.aw_l1, true}, {&w.aw_l2, true}, {&w.aw_l3, false}});

    // --- attention values: [E, node] ---
    auto av_in = concat_rows(E, {{&s_src, node}, {&z, pair}});
    auto attn_value = dense_stack(av_in, E, node + pair,
                                  {{&w.av_l1, true}, {&w.av_l2, true}, {&w.av_l3, false}});

    // softmax over destination groups (per head), then weighted aggregate.
    attn_weight = scatter_softmax(attn_weight, E, heads, dst, num_nodes);

    // attn_output[e, h*node + c] = attn_weight[e,h] * attn_value[e,c]
    std::vector<float> attn_output((size_t)E * heads * node);
    for (int e = 0; e < E; ++e)
        for (int h = 0; h < heads; ++h)
            for (int c = 0; c < node; ++c)
                attn_output[e * heads * node + h * node + c] =
                    attn_weight[e * heads + h] * attn_value[e * node + c];

    auto agg = scatter_sum(attn_output, E, heads * node, dst, num_nodes);  // [N, heads*node]

    // --- s = s + BN(out_linear(agg)) ---
    {
        auto out = dense_stack(agg, num_nodes, heads * node, {{&w.out_l, false}});  // [N, node]
        apply_affine(out, num_nodes, node, w.out_bn);
        for (int i = 0; i < num_nodes * node; ++i) s[i] += out[i];
    }

    // --- s = s + BN(FFN(s)) ---
    {
        auto out = dense_stack(s, num_nodes, node, {{&w.ffn_l1, true}, {&w.ffn_l2, false}});
        apply_affine(out, num_nodes, node, w.ffn_bn);
        for (int i = 0; i < num_nodes * node; ++i) s[i] += out[i];
    }

    return {std::move(s), std::move(z)};
}

std::vector<float> linear_batch(const Linear& L, const std::vector<float>& x, int rows) {
    return dense_stack(x, rows, L.in, {{&L, false}});
}

std::vector<float> mlp_attn_gnn_decoder_forward(const MLPAttnGNNWeights& w, int num_nodes,
                                                const std::vector<float>& s_in,
                                                const std::vector<float>& nbr,
                                                const std::vector<int>& src,
                                                const std::vector<int>& dst,
                                                int neighbor_dim) {
    const int node = w.node_dim, heads = w.num_heads;
    const int E = static_cast<int>(src.size());

    std::vector<float> s = s_in;  // [N, node]
    const std::vector<float> s_dst = gather(s, node, dst);  // [E, node] (current s)

    // attn_weight = attn_weight_mlp(cat[s[dst], nbr]) -> [E, heads]
    auto aw_in = concat_rows(E, {{&s_dst, node}, {&nbr, neighbor_dim}});
    auto attn_weight = dense_stack(aw_in, E, node + neighbor_dim,
                                   {{&w.aw_l1, true}, {&w.aw_l2, true}, {&w.aw_l3, false}});

    // attn_value = attn_value_mlp(nbr) -> [E, node]
    auto attn_value = dense_stack(nbr, E, neighbor_dim,
                                  {{&w.av_l1, true}, {&w.av_l2, true}, {&w.av_l3, false}});

    attn_weight = scatter_softmax(attn_weight, E, heads, dst, num_nodes);

    std::vector<float> attn_output((size_t)E * heads * node);
    for (int e = 0; e < E; ++e)
        for (int h = 0; h < heads; ++h)
            for (int c = 0; c < node; ++c)
                attn_output[e * heads * node + h * node + c] =
                    attn_weight[e * heads + h] * attn_value[e * node + c];

    auto agg = scatter_sum(attn_output, E, heads * node, dst, num_nodes);  // [N, heads*node]

    // s = s + BN(out_linear(agg))
    {
        auto out = dense_stack(agg, num_nodes, heads * node, {{&w.out_l, false}});
        apply_affine(out, num_nodes, node, w.out_bn);
        for (int i = 0; i < num_nodes * node; ++i) s[i] += out[i];
    }
    // s = s + BN(FFN(s))
    {
        auto out = dense_stack(s, num_nodes, node, {{&w.ffn_l1, true}, {&w.ffn_l2, false}});
        apply_affine(out, num_nodes, node, w.ffn_bn);
        for (int i = 0; i < num_nodes * node; ++i) s[i] += out[i];
    }
    return s;
}

}  // namespace boltz
