#include "boltz/ifold_real.hpp"

#include "boltz/gnn.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace boltz {
namespace {

// Linear from "<name>.weight" ([out,in] torch -> ggml ne=[in,out]) and an
// optional "<name>.bias"; bias defaults to zeros when absent (e.g. predictor).
Linear load_linear(const WeightStore& ws, const std::string& name) {
    Linear L;
    ggml_tensor* t = ws.get(name + ".weight");
    L.in = static_cast<int>(t->ne[0]);
    L.out = static_cast<int>(t->ne[1]);
    L.w = ws.data_f32(name + ".weight");
    if (ws.has(name + ".bias"))
        L.b = ws.data_f32(name + ".bias");
    else
        L.b.assign(L.out, 0.0f);
    return L;
}

// Fold a BatchNorm1d/SyncBatchNorm (eval) into y = x*scale + shift:
//   scale = weight / sqrt(running_var + eps),  shift = bias - running_mean*scale.
Affine fold_bn(const WeightStore& ws, const std::string& name, float eps = 1e-5f) {
    const std::vector<float> g = ws.data_f32(name + ".weight");
    const std::vector<float> b = ws.data_f32(name + ".bias");
    const std::vector<float> rm = ws.data_f32(name + ".running_mean");
    const std::vector<float> rv = ws.data_f32(name + ".running_var");
    const int F = static_cast<int>(g.size());
    Affine a;
    a.scale.resize(F);
    a.shift.resize(F);
    for (int c = 0; c < F; ++c) {
        a.scale[c] = g[c] / std::sqrt(rv[c] + eps);
        a.shift[c] = b[c] - rm[c] * a.scale[c];
    }
    return a;
}

// Load one MLPAttnGNN(-decoder) layer. `with_edge` loads the encoder-only
// edge_FFN (the decoder has none).
MLPAttnGNNWeights load_layer(const WeightStore& ws, const std::string& p,
                             const RealIFoldConfig& c, bool with_edge) {
    MLPAttnGNNWeights w;
    w.node_dim = c.node_dim;
    w.pair_dim = c.pair_dim;
    w.hidden_dim = c.node_dim;
    w.num_heads = c.num_heads;

    w.aw_l1 = load_linear(ws, p + "attn_weight_mlp.0");
    w.aw_l2 = load_linear(ws, p + "attn_weight_mlp.2");
    w.aw_l3 = load_linear(ws, p + "attn_weight_mlp.4");

    w.av_l1 = load_linear(ws, p + "attn_value_mlp.0");
    w.av_l2 = load_linear(ws, p + "attn_value_mlp.2");
    w.av_l3 = load_linear(ws, p + "attn_value_mlp.4");

    w.out_l = load_linear(ws, p + "attn_output_linear.0");
    w.out_bn = fold_bn(ws, p + "attn_output_linear.2");

    w.ffn_l1 = load_linear(ws, p + "attn_FFN.0");
    w.ffn_l2 = load_linear(ws, p + "attn_FFN.2");
    w.ffn_bn = fold_bn(ws, p + "attn_FFN.4");

    if (with_edge) {
        w.edge_l1 = load_linear(ws, p + "edge_FFN.0");
        w.edge_l2 = load_linear(ws, p + "edge_FFN.2");
        w.edge_bn = fold_bn(ws, p + "edge_FFN.4");
    }
    return w;
}

int count_layers(const WeightStore& ws, const std::string& prefix) {
    int n = 0;
    while (ws.has(prefix + std::to_string(n) + ".attn_weight_mlp.0.weight")) ++n;
    return n;
}

}  // namespace

RealIFoldModel load_real_ifold_model(const WeightStore& ws) {
    RealIFoldModel m;
    RealIFoldConfig& c = m.cfg;

    // Short prefixes emitted by convert_ckpt_to_gguf.py --arch boltzgen-ifold
    // (the verbatim "inverse_folding_encoder."/"structure_module." prefixes would
    // push tensor names past ggml's 64-char limit).
    const std::string ep = "enc.";
    const std::string dp = "dec.";

    m.linear_token_to_node = load_linear(ws, ep + "linear_token_to_node");
    m.linear_token_to_pair = load_linear(ws, ep + "linear_token_to_pair");

    c.node_dim = m.linear_token_to_node.out;
    c.token_s = m.linear_token_to_node.in;
    c.pair_dim = m.linear_token_to_pair.out;
    c.edge_in = m.linear_token_to_pair.in;
    // num_heads = output width of attn_weight_mlp's last linear.
    c.num_heads = static_cast<int>(ws.get(ep + "encoder_layers.0.attn_weight_mlp.4.weight")->ne[1]);
    c.n_enc = count_layers(ws, ep + "encoder_layers.");
    c.n_dec = count_layers(ws, dp + "decoder_layers.");
    c.neighbor_dim = c.pair_dim + c.node_dim;

    if (c.n_enc == 0) throw std::runtime_error("ifold gguf: no encoder layers found");

    for (int l = 0; l < c.n_enc; ++l)
        m.enc.push_back(load_layer(ws, ep + "encoder_layers." + std::to_string(l) + ".", c, true));
    for (int l = 0; l < c.n_dec; ++l)
        m.dec.push_back(load_layer(ws, dp + "decoder_layers." + std::to_string(l) + ".", c, false));

    m.seq_to_s = load_linear(ws, dp + "seq_to_s");
    m.predictor = load_linear(ws, dp + "predictor");
    c.num_tokens = m.predictor.out;
    return m;
}

RealEncoderOut run_real_encoder(const RealIFoldModel& m,
                                const std::vector<float>& s_inputs,
                                const std::vector<float>& pair_input,
                                const std::vector<int>& src,
                                const std::vector<int>& dst, int N) {
    const RealIFoldConfig& c = m.cfg;
    std::vector<float> s = linear_batch(m.linear_token_to_node, s_inputs, N);
    const int E = static_cast<int>(src.size());
    std::vector<float> z = linear_batch(m.linear_token_to_pair, pair_input, E);
    for (const MLPAttnGNNWeights& lw : m.enc) {
        MLPAttnGNNState st = mlp_attn_gnn_forward(lw, N, s, z, src, dst);
        s = std::move(st.s);
        z = std::move(st.z);
    }
    (void)c;
    return {std::move(s), std::move(z)};
}

std::vector<float> run_real_decoder_logits(const RealIFoldModel& m,
                                           const std::vector<float>& s_enc,
                                           const std::vector<float>& z_enc,
                                           const std::vector<float>& res_type_vis,
                                           const std::vector<int>& src,
                                           const std::vector<int>& dst, int N) {
    const RealIFoldConfig& c = m.cfg;
    const int E = static_cast<int>(src.size());
    const int node = c.node_dim, pair = c.pair_dim, nbrd = c.neighbor_dim;

    // res_rep = seq_to_s(res_type_vis)  -> [E, node]
    const std::vector<float> res_rep = linear_batch(m.seq_to_s, res_type_vis, E);

    // neighbors_rep = cat([z_enc, s_enc[src] + res_rep], -1)  -> [E, pair+node]
    std::vector<float> nbr((size_t)E * nbrd);
    for (int e = 0; e < E; ++e) {
        const int a = src[e];
        for (int p = 0; p < pair; ++p) nbr[e * nbrd + p] = z_enc[e * pair + p];
        for (int d = 0; d < node; ++d)
            nbr[e * nbrd + pair + d] = s_enc[a * node + d] + res_rep[e * node + d];
    }

    std::vector<float> s = s_enc;
    for (const MLPAttnGNNWeights& lw : m.dec)
        s = mlp_attn_gnn_decoder_forward(lw, N, s, nbr, src, dst, nbrd);

    return linear_batch(m.predictor, s, N);  // [N, num_tokens]
}

}  // namespace boltz
