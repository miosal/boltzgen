#include "boltz/ifold_model.hpp"

#include "boltz/const.hpp"
#include "boltz/featurizer.hpp"
#include "boltz/gnn.hpp"

#include <cmath>
#include <stdexcept>

namespace boltz {
namespace {

// Canonical one-letter codes in canonical_tokens order.
const char* kOneLetter = "ARNDCQEGHILKMFPSTWYV";  // 20

Linear load_linear(const WeightStore& ws, const std::string& name) {
    Linear L;
    ggml_tensor* t = ws.get(name + ".weight");
    L.in = static_cast<int>(t->ne[0]);
    L.out = static_cast<int>(t->ne[1]);
    L.w = ws.data_f32(name + ".weight");
    L.b = ws.data_f32(name + ".bias");
    return L;
}

Affine load_affine(const WeightStore& ws, const std::string& name) {
    Affine a;
    a.scale = ws.data_f32(name + ".scale");
    a.shift = ws.data_f32(name + ".shift");
    return a;
}

MLPAttnGNNWeights load_layer(const WeightStore& ws, const std::string& p,
                             const IFoldConfig& c) {
    MLPAttnGNNWeights w;
    w.node_dim = c.node_dim;
    w.pair_dim = c.pair_dim;
    w.hidden_dim = c.hidden_dim;
    w.num_heads = c.num_heads;
    w.edge_l1 = load_linear(ws, p + "edge_l1");
    w.edge_l2 = load_linear(ws, p + "edge_l2");
    w.edge_bn = load_affine(ws, p + "edge_bn");
    w.aw_l1 = load_linear(ws, p + "aw_l1");
    w.aw_l2 = load_linear(ws, p + "aw_l2");
    w.aw_l3 = load_linear(ws, p + "aw_l3");
    w.av_l1 = load_linear(ws, p + "av_l1");
    w.av_l2 = load_linear(ws, p + "av_l2");
    w.av_l3 = load_linear(ws, p + "av_l3");
    w.out_l = load_linear(ws, p + "out_l");
    w.out_bn = load_affine(ws, p + "out_bn");
    w.ffn_l1 = load_linear(ws, p + "ffn_l1");
    w.ffn_l2 = load_linear(ws, p + "ffn_l2");
    w.ffn_bn = load_affine(ws, p + "ffn_bn");
    return w;
}

int meta_int(const WeightStore& ws, const std::string& key) {
    auto v = ws.meta_u32(key);
    if (!v.has_value()) throw std::runtime_error("ifold gguf missing meta '" + key + "'");
    return static_cast<int>(*v);
}

std::vector<float> linear_apply(const std::vector<float>& x, const Linear& L) {
    std::vector<float> y(L.out);
    for (int o = 0; o < L.out; ++o) {
        float s = L.b[o];
        for (int i = 0; i < L.in; ++i) s += L.w[o * L.in + i] * x[i];
        y[o] = s;
    }
    return y;
}

}  // namespace

IFoldModel load_ifold_model(const WeightStore& ws) {
    IFoldModel m;
    IFoldConfig& c = m.cfg;
    c.node_dim = meta_int(ws, "ifold.node_dim");
    c.pair_dim = meta_int(ws, "ifold.pair_dim");
    c.hidden_dim = meta_int(ws, "ifold.hidden_dim");
    c.num_heads = meta_int(ws, "ifold.num_heads");
    c.num_layers = meta_int(ws, "ifold.num_layers");
    c.num_gaussians = meta_int(ws, "ifold.num_gaussians");
    c.topk = meta_int(ws, "ifold.topk");
    c.num_embed = meta_int(ws, "ifold.num_embed");

    m.w.tok_embed = ws.data_f32("tok_embed.weight");
    m.w.edge_proj = load_linear(ws, "edge_proj");
    for (int l = 0; l < c.num_layers; ++l)
        m.w.layers.push_back(load_layer(ws, "enc." + std::to_string(l) + ".", c));
    m.w.head_ln_w = ws.data_f32("head_norm.weight");
    m.w.head_ln_b = ws.data_f32("head_norm.bias");
    m.w.head = load_linear(ws, "head");
    return m;
}

IFoldResult run_ifold(const IFoldModel& m, const std::vector<Residue>& residues,
                      const std::vector<char>& design_mask) {
    const IFoldConfig& c = m.cfg;
    const int N = static_cast<int>(residues.size());
    const int node = c.node_dim, pair = c.pair_dim;

    // Coordinates + validity from CA atoms.
    std::vector<float> coords(static_cast<size_t>(N) * 3);
    std::vector<char> valid(N);
    for (int i = 0; i < N; ++i) {
        coords[i * 3 + 0] = residues[i].ca_x;
        coords[i * 3 + 1] = residues[i].ca_y;
        coords[i * 3 + 2] = residues[i].ca_z;
        valid[i] = residues[i].has_ca ? 1 : 0;
    }

    KnnGraph g = knn_graph(coords, valid, N, c.topk);

    // Node init: embed (masked) residue identity. Designed positions -> mask
    // token (index num_embed-1); otherwise the canonical token, falling back to
    // mask for non-standard residues.
    const int mask_tok = c.num_embed - 1;
    std::vector<float> s(static_cast<size_t>(N) * node);
    for (int i = 0; i < N; ++i) {
        int tok = mask_tok;
        if (!(design_mask.size() == static_cast<size_t>(N) && design_mask[i])) {
            const int ci = canonical_index(residues[i].comp_id);
            tok = (ci >= 0) ? ci : mask_tok;
        }
        for (int d = 0; d < node; ++d) s[i * node + d] = m.w.tok_embed[tok * node + d];
    }

    // Edge init: gaussian-smeared CA-CA distance projected to pair_dim.
    const int E = static_cast<int>(g.src_idx.size());
    std::vector<float> z(static_cast<size_t>(E) * pair);
    for (int e = 0; e < E; ++e) {
        const int a = g.src_idx[e], b = g.dst_idx[e];
        const float dx = coords[a * 3] - coords[b * 3];
        const float dy = coords[a * 3 + 1] - coords[b * 3 + 1];
        const float dz = coords[a * 3 + 2] - coords[b * 3 + 2];
        const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
        Matrix sm = gaussian_smearing({d}, c.g_start, c.g_stop, c.num_gaussians);
        std::vector<float> proj = linear_apply(sm.data, m.w.edge_proj);  // [pair]
        for (int p = 0; p < pair; ++p) z[e * pair + p] = proj[p];
    }

    // Encoder stack (each layer runs its dense MLPs through ggml).
    for (const MLPAttnGNNWeights& lw : m.w.layers) {
        MLPAttnGNNState st = mlp_attn_gnn_forward(lw, N, s, z, g.src_idx, g.dst_idx);
        s = std::move(st.s);
        z = std::move(st.z);
    }

    // Head: LayerNorm(node) + Linear(node->20), argmax per residue.
    IFoldResult r;
    r.tokens.resize(N);
    r.sequence.resize(N);
    for (int i = 0; i < N; ++i) {
        std::vector<float> h(node);
        float mean = 0;
        for (int d = 0; d < node; ++d) mean += s[i * node + d];
        mean /= node;
        float var = 0;
        for (int d = 0; d < node; ++d) {
            const float v = s[i * node + d] - mean;
            var += v * v;
        }
        var /= node;
        const float inv = 1.0f / std::sqrt(var + 1e-5f);
        for (int d = 0; d < node; ++d)
            h[d] = (s[i * node + d] - mean) * inv * m.w.head_ln_w[d] + m.w.head_ln_b[d];

        std::vector<float> logits = linear_apply(h, m.w.head);  // [20]
        int best = 0;
        for (int k = 1; k < 20; ++k) if (logits[k] > logits[best]) best = k;
        r.tokens[i] = best;
        r.sequence[i] = kOneLetter[best];
    }
    return r;
}

}  // namespace boltz
