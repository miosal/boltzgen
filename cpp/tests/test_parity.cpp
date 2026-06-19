// Numeric-parity test against the PyTorch reference on REAL trained weights.
//
// Loads the converted inverse-fold checkpoint (boltzgen1_ifold.gguf) plus golden
// fixtures dumped from the reference model (tools/dump_golden.py), runs the C++
// inverse_folding_encoder + structure_module(decoder) forward, and asserts the
// outputs match the reference within tolerance. This closes Phase-1 numeric
// parity for the inverse-fold GNN core (encoder + decoder, 314/357 checkpoint
// tensors) on trained weights.
//
// The gguf + fixtures are derived from a HuggingFace-gated checkpoint and live
// under the build dir (not committed); regenerate with:
//   python3 cpp/tools/convert_ckpt_to_gguf.py boltzgen1_ifold.ckpt \
//       cpp/build/boltzgen1_ifold.gguf --arch boltzgen-ifold
//   python3 cpp/tools/dump_golden.py boltzgen1_ifold.ckpt cpp/build/golden_ifold
// If they are absent the test SKIPS (data-gated), like the other example tests.
#include "boltz/attention_pair_bias.hpp"
#include "boltz/ifold_real.hpp"
#include "boltz/npy.hpp"
#include "boltz/triangle_mult.hpp"
#include "boltz/weight_store.hpp"
#include "test_framework.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace boltz;
using namespace boltztest;

#ifndef BOLTZ_PARITY_GGUF
#define BOLTZ_PARITY_GGUF "boltzgen1_ifold.gguf"
#endif
#ifndef BOLTZ_PARITY_DIR
#define BOLTZ_PARITY_DIR "golden_ifold"
#endif
#ifndef BOLTZ_PARITY_DESIGN_GGUF
#define BOLTZ_PARITY_DESIGN_GGUF "boltzgen1_design_layer0.gguf"
#endif
#ifndef BOLTZ_PARITY_TRIMUL_DIR
#define BOLTZ_PARITY_TRIMUL_DIR "golden_trimul"
#endif
#ifndef BOLTZ_PARITY_ATTN_DIR
#define BOLTZ_PARITY_ATTN_DIR "golden_attn"
#endif

namespace {

bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return f.good();
}

std::vector<int> to_int(const NpyArray& a) {
    std::vector<int> v(a.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) v[i] = static_cast<int>(a.data[i] + 0.5f);
    return v;
}

NpyArray wrap(const std::vector<float>& d, std::vector<long> shape) {
    NpyArray a;
    a.data = d;
    a.shape = std::move(shape);
    return a;
}

}  // namespace

BOLTZ_TEST(ifold_real_weights_parity) {
    const std::string gguf = BOLTZ_PARITY_GGUF;
    const std::string dir = BOLTZ_PARITY_DIR;
    if (!file_exists(gguf) || !file_exists(dir + "/s_inputs.npy")) {
        std::printf("    (skip: data-gated — gguf/fixtures absent; see header)\n");
        return;
    }

    WeightStore ws(gguf);
    RealIFoldModel m = load_real_ifold_model(ws);
    std::printf("    model: node=%d pair=%d token_s=%d edge_in=%d heads=%d enc=%d dec=%d ntok=%d\n",
                m.cfg.node_dim, m.cfg.pair_dim, m.cfg.token_s, m.cfg.edge_in, m.cfg.num_heads,
                m.cfg.n_enc, m.cfg.n_dec, m.cfg.num_tokens);

    // --- load fixtures ---
    NpyArray s_inputs = load_npy(dir + "/s_inputs.npy");
    NpyArray pair_input = load_npy(dir + "/pair_input.npy");
    NpyArray res_type_vis = load_npy(dir + "/res_type_vis.npy");
    NpyArray s_enc_g = load_npy(dir + "/s_enc.npy");
    NpyArray z_enc_g = load_npy(dir + "/z_enc.npy");
    NpyArray logits_g = load_npy(dir + "/logits.npy");
    std::vector<int> src = to_int(load_npy(dir + "/edge_src.npy"));
    std::vector<int> dst = to_int(load_npy(dir + "/edge_dst.npy"));

    const int N = static_cast<int>(s_inputs.shape[0]);
    const int E = static_cast<int>(src.size());
    std::printf("    scenario: N=%d E=%d\n", N, E);

    // --- encoder parity ---
    RealEncoderOut enc = run_real_encoder(m, s_inputs.data, pair_input.data, src, dst, N);
    DiffStats ds = compare(wrap(enc.s, {N, m.cfg.node_dim}), s_enc_g);
    DiffStats dz = compare(wrap(enc.z, {E, m.cfg.pair_dim}), z_enc_g);
    std::printf("    encoder s: max_abs=%.3e mean_abs=%.3e (ref std~%.2f)\n",
                ds.max_abs, ds.mean_abs, 4.03);
    std::printf("    encoder z: max_abs=%.3e mean_abs=%.3e (ref std~%.2f)\n",
                dz.max_abs, dz.mean_abs, 5.09);
    expect_true(ds.shapes_match, "encoder s shape");
    expect_true(dz.shapes_match, "encoder z shape");
    expect_true(ds.max_abs < 2e-2f, "encoder s parity (max_abs < 2e-2)");
    expect_true(dz.max_abs < 2e-2f, "encoder z parity (max_abs < 2e-2)");

    // --- decoder parity (uses the reference encoder outputs as inputs) ---
    std::vector<float> logits =
        run_real_decoder_logits(m, s_enc_g.data, z_enc_g.data, res_type_vis.data, src, dst, N);
    DiffStats dl = compare(wrap(logits, {N, m.cfg.num_tokens}), logits_g);
    // relative to the logits' magnitude (ref std ~86).
    float maxabs_ref = 0.0f;
    for (float v : logits_g.data) maxabs_ref = std::max(maxabs_ref, std::fabs(v));
    std::printf("    decoder logits: max_abs=%.3e mean_abs=%.3e (ref max|v|=%.2f -> rel=%.2e)\n",
                dl.max_abs, dl.mean_abs, maxabs_ref, dl.max_abs / (maxabs_ref + 1e-9f));
    expect_true(dl.shapes_match, "decoder logits shape");
    expect_true(dl.max_abs < 5e-1f, "decoder logits parity (max_abs < 5e-1)");
    expect_true(dl.max_abs / (maxabs_ref + 1e-9f) < 1e-3f, "decoder logits rel parity (< 1e-3)");

    // argmax (greedy designed token) must match exactly per position.
    int argmax_mismatch = 0;
    for (int i = 0; i < N; ++i) {
        int gb = 0, cb = 0;
        for (int k = 1; k < m.cfg.num_tokens; ++k) {
            if (logits_g.data[i * m.cfg.num_tokens + k] > logits_g.data[i * m.cfg.num_tokens + gb]) gb = k;
            if (logits[i * m.cfg.num_tokens + k] > logits[i * m.cfg.num_tokens + cb]) cb = k;
        }
        if (gb != cb) ++argmax_mismatch;
    }
    std::printf("    decoder argmax mismatches: %d / %d positions\n", argmax_mismatch, N);
    expect_eq_i(argmax_mismatch, 0, "decoder per-position argmax matches reference");
}

namespace {
TriMulWeights load_trimul(const WeightStore& ws, const std::string& p) {
    TriMulWeights w;
    w.norm_in_w = ws.data_f32(p + "norm_in.weight");
    w.norm_in_b = ws.data_f32(p + "norm_in.bias");
    w.p_in = ws.data_f32(p + "p_in.weight");
    w.g_in = ws.data_f32(p + "g_in.weight");
    w.norm_out_w = ws.data_f32(p + "norm_out.weight");
    w.norm_out_b = ws.data_f32(p + "norm_out.bias");
    w.p_out = ws.data_f32(p + "p_out.weight");
    w.g_out = ws.data_f32(p + "g_out.weight");
    w.dim = static_cast<int>(w.norm_in_w.size());
    return w;
}
}  // namespace

// Triangle-multiplication kernel parity on REAL design-checkpoint weights — the
// signature Pairformer pair op, shared by the design/folding/affinity trunks.
// Demonstrates the convert->dump->parity flow extending to a 2 GB checkpoint
// (one block extracted via convert_ckpt_to_gguf.py --include).
BOLTZ_TEST(trimul_real_weights_parity) {
    const std::string gguf = BOLTZ_PARITY_DESIGN_GGUF;
    const std::string dir = BOLTZ_PARITY_TRIMUL_DIR;
    if (!file_exists(gguf) || !file_exists(dir + "/trimul_x.npy")) {
        std::printf("    (skip: data-gated — design gguf/fixtures absent; see header)\n");
        return;
    }
    WeightStore ws(gguf);
    const std::string base = "pairformer_module.layers.0.";
    TriMulWeights w_out = load_trimul(ws, base + "tri_mul_out.");
    TriMulWeights w_in = load_trimul(ws, base + "tri_mul_in.");

    NpyArray x = load_npy(dir + "/trimul_x.npy");           // [N, N, D]
    NpyArray mask = load_npy(dir + "/trimul_mask.npy");     // [N, N]
    NpyArray go = load_npy(dir + "/trimul_out_outgoing.npy");
    NpyArray gi = load_npy(dir + "/trimul_out_incoming.npy");
    const int N = static_cast<int>(x.shape[0]);
    const int D = static_cast<int>(x.shape[2]);
    std::printf("    trimul: N=%d D=%d\n", N, D);

    std::vector<float> yo = triangle_multiplication(x.data, mask.data, N, D, w_out, false);
    std::vector<float> yi = triangle_multiplication(x.data, mask.data, N, D, w_in, true);
    DiffStats do_ = compare(wrap(yo, {N, N, D}), go);
    DiffStats di_ = compare(wrap(yi, {N, N, D}), gi);
    std::printf("    tri_mul_out: max_abs=%.3e mean_abs=%.3e\n", do_.max_abs, do_.mean_abs);
    std::printf("    tri_mul_in : max_abs=%.3e mean_abs=%.3e\n", di_.max_abs, di_.mean_abs);
    expect_true(do_.shapes_match && di_.shapes_match, "trimul shapes");
    expect_true(do_.max_abs < 2e-3f, "tri_mul outgoing parity (max_abs < 2e-3)");
    expect_true(di_.max_abs < 2e-3f, "tri_mul incoming parity (max_abs < 2e-3)");
}

namespace {
AttnPairBiasWeights load_attn(const WeightStore& ws, const std::string& p) {
    AttnPairBiasWeights w;
    w.c_s = static_cast<int>(ws.get(p + "proj_q.weight")->ne[1]);
    ggml_tensor* zl = ws.get(p + "proj_z.1.weight");  // ne=[c_z, num_heads]
    w.c_z = static_cast<int>(zl->ne[0]);
    w.num_heads = static_cast<int>(zl->ne[1]);
    w.compute_pair_bias = true;
    w.inf = 1e6f;
    w.q_w = ws.data_f32(p + "proj_q.weight");
    w.q_b = ws.data_f32(p + "proj_q.bias");
    w.k_w = ws.data_f32(p + "proj_k.weight");
    w.v_w = ws.data_f32(p + "proj_v.weight");
    w.g_w = ws.data_f32(p + "proj_g.weight");
    w.z_norm_w = ws.data_f32(p + "proj_z.0.weight");
    w.z_norm_b = ws.data_f32(p + "proj_z.0.bias");
    w.z_lin_w = ws.data_f32(p + "proj_z.1.weight");
    w.o_w = ws.data_f32(p + "proj_o.weight");
    return w;
}
}  // namespace

// Pair-bias attention kernel parity on REAL design-checkpoint weights — the
// single-rep attention used across the Pairformer trunk and diffusion
// transformer. All-valid mask so the core attention+bias+gate+output-proj is
// what's compared (the C++ counterpart is boltz::attention_pair_bias).
BOLTZ_TEST(attn_pair_bias_real_weights_parity) {
    const std::string gguf = BOLTZ_PARITY_DESIGN_GGUF;
    const std::string dir = BOLTZ_PARITY_ATTN_DIR;
    if (!file_exists(gguf) || !file_exists(dir + "/attn_s.npy")) {
        std::printf("    (skip: data-gated — design gguf/fixtures absent; see header)\n");
        return;
    }
    WeightStore ws(gguf);
    AttnPairBiasWeights w = load_attn(ws, "pairformer_module.layers.0.attention.");
    std::printf("    attn: c_s=%d c_z=%d heads=%d\n", w.c_s, w.c_z, w.num_heads);

    NpyArray s = load_npy(dir + "/attn_s.npy");     // [N, c_s]
    NpyArray z = load_npy(dir + "/attn_z.npy");     // [N, N, c_z]
    NpyArray go = load_npy(dir + "/attn_out.npy");  // [N, c_s]
    const int N = static_cast<int>(s.shape[0]);
    std::vector<float> mask((size_t)N * N, 1.0f);   // all-valid

    std::vector<float> out = attention_pair_bias(s.data, z.data, mask, N, w);
    DiffStats d = compare(wrap(out, {N, w.c_s}), go);
    float maxabs_ref = 0.0f;
    for (float v : go.data) maxabs_ref = std::max(maxabs_ref, std::fabs(v));
    std::printf("    attn out: max_abs=%.3e mean_abs=%.3e (ref max|v|=%.2f -> rel=%.2e)\n",
                d.max_abs, d.mean_abs, maxabs_ref, d.max_abs / (maxabs_ref + 1e-9f));
    expect_true(d.shapes_match, "attn out shape");
    expect_true(d.max_abs < 5e-3f, "attn pair-bias parity (max_abs < 5e-3)");
}

int main() {
    std::printf("== test_parity ==\n");
    return boltztest::run_all();
}
