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
#include "boltz/ifold_real.hpp"
#include "boltz/npy.hpp"
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

int main() {
    std::printf("== test_parity ==\n");
    return boltztest::run_all();
}
