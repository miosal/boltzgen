// End-to-end pipeline test for the inverse-folding model: build a tiny synthetic
// gguf (all tensors + metadata) in C++, load it, and run the full path
// (residues -> KNN -> embed -> ggml encoder stack -> head -> sequence). No
// python, no trained weights — validates that the assembled pipeline executes
// and produces well-formed output.
#include "boltz/cif.hpp"
#include "boltz/ifold_model.hpp"
#include "boltz/weight_store.hpp"
#include "test_framework.hpp"

#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

// Tiny config.
constexpr int NODE = 4, PAIR = 4, HID = 4, HEADS = 2, LAYERS = 2, GAUSS = 4,
              TOPK = 3, NEMBED = 21;

float fill_value(const std::string& name, int i) {
    if (name.size() >= 6 && name.substr(name.size() - 6) == ".scale") return 1.0f;
    if (name.size() >= 6 && name.substr(name.size() - 6) == ".shift") return 0.0f;
    if (name == "head_norm.weight") return 1.0f;
    if (name == "head_norm.bias") return 0.0f;
    return 0.02f * std::sin(static_cast<float>((std::hash<std::string>{}(name) % 97) + i) * 0.3f);
}

ggml_tensor* add_tensor(ggml_context* ctx, gguf_context* g, const std::string& name,
                        const std::vector<int>& ne) {
    ggml_tensor* t;
    if (ne.size() == 1) t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
    else t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
    ggml_set_name(t, name.c_str());
    float* d = ggml_get_data_f32(t);
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) d[i] = fill_value(name, static_cast<int>(i));
    gguf_add_tensor(g, t);
    return t;
}

void add_layer(ggml_context* ctx, gguf_context* g, const std::string& p) {
    add_tensor(ctx, g, p + "edge_l1.weight", {2 * NODE + PAIR, HID});
    add_tensor(ctx, g, p + "edge_l1.bias", {HID});
    add_tensor(ctx, g, p + "edge_l2.weight", {HID, PAIR});
    add_tensor(ctx, g, p + "edge_l2.bias", {PAIR});
    add_tensor(ctx, g, p + "edge_bn.scale", {PAIR});
    add_tensor(ctx, g, p + "edge_bn.shift", {PAIR});
    add_tensor(ctx, g, p + "aw_l1.weight", {2 * NODE + PAIR, HID});
    add_tensor(ctx, g, p + "aw_l1.bias", {HID});
    add_tensor(ctx, g, p + "aw_l2.weight", {HID, HID});
    add_tensor(ctx, g, p + "aw_l2.bias", {HID});
    add_tensor(ctx, g, p + "aw_l3.weight", {HID, HEADS});
    add_tensor(ctx, g, p + "aw_l3.bias", {HEADS});
    add_tensor(ctx, g, p + "av_l1.weight", {NODE + PAIR, HID});
    add_tensor(ctx, g, p + "av_l1.bias", {HID});
    add_tensor(ctx, g, p + "av_l2.weight", {HID, HID});
    add_tensor(ctx, g, p + "av_l2.bias", {HID});
    add_tensor(ctx, g, p + "av_l3.weight", {HID, NODE});
    add_tensor(ctx, g, p + "av_l3.bias", {NODE});
    add_tensor(ctx, g, p + "out_l.weight", {HEADS * NODE, NODE});
    add_tensor(ctx, g, p + "out_l.bias", {NODE});
    add_tensor(ctx, g, p + "out_bn.scale", {NODE});
    add_tensor(ctx, g, p + "out_bn.shift", {NODE});
    add_tensor(ctx, g, p + "ffn_l1.weight", {NODE, HID});
    add_tensor(ctx, g, p + "ffn_l1.bias", {HID});
    add_tensor(ctx, g, p + "ffn_l2.weight", {HID, NODE});
    add_tensor(ctx, g, p + "ffn_l2.bias", {NODE});
    add_tensor(ctx, g, p + "ffn_bn.scale", {NODE});
    add_tensor(ctx, g, p + "ffn_bn.shift", {NODE});
}

std::string write_synth_gguf() {
    const std::string path = "/tmp/boltzcpp_ifold_synth.gguf";
    ggml_init_params ip{(size_t)64 * 1024 * 1024, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    gguf_context* g = gguf_init_empty();

    gguf_set_val_str(g, "general.architecture", "boltzgen-ifold");
    gguf_set_val_u32(g, "ifold.node_dim", NODE);
    gguf_set_val_u32(g, "ifold.pair_dim", PAIR);
    gguf_set_val_u32(g, "ifold.hidden_dim", HID);
    gguf_set_val_u32(g, "ifold.num_heads", HEADS);
    gguf_set_val_u32(g, "ifold.num_layers", LAYERS);
    gguf_set_val_u32(g, "ifold.num_gaussians", GAUSS);
    gguf_set_val_u32(g, "ifold.topk", TOPK);
    gguf_set_val_u32(g, "ifold.num_embed", NEMBED);

    add_tensor(ctx, g, "tok_embed.weight", {NODE, NEMBED});
    add_tensor(ctx, g, "edge_proj.weight", {GAUSS, PAIR});
    add_tensor(ctx, g, "edge_proj.bias", {PAIR});
    for (int l = 0; l < LAYERS; ++l) add_layer(ctx, g, "enc." + std::to_string(l) + ".");
    add_tensor(ctx, g, "head_norm.weight", {NODE});
    add_tensor(ctx, g, "head_norm.bias", {NODE});
    add_tensor(ctx, g, "head.weight", {NODE, 20});
    add_tensor(ctx, g, "head.bias", {20});

    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

Residue res(const char* comp, int seq, float x, float y, float z) {
    Residue r;
    r.asym_id = "A";
    r.comp_id = comp;
    r.seq_id = seq;
    r.ca_x = x; r.ca_y = y; r.ca_z = z;
    r.has_ca = true;
    return r;
}

}  // namespace

BOLTZ_TEST(ifold_end_to_end_runs) {
    const std::string path = write_synth_gguf();
    WeightStore ws(path);
    IFoldModel m = load_ifold_model(ws);

    expect_eq_i(m.cfg.num_layers, LAYERS, "layers loaded");
    expect_eq_i(static_cast<long>(m.w.layers.size()), LAYERS, "layer weights");

    std::vector<Residue> residues = {
        res("VAL", 1, 0, 0, 0), res("ILE", 2, 3.8f, 0, 0),
        res("ASN", 3, 7.6f, 0, 0), res("GLY", 4, 7.6f, 3.8f, 0),
        res("LEU", 5, 3.8f, 3.8f, 0), res("SER", 6, 0, 3.8f, 0),
    };
    const int N = static_cast<int>(residues.size());
    std::vector<char> design_mask(N, 0);
    design_mask[2] = 1;  // design residue 3

    IFoldResult r = run_ifold(m, residues, design_mask);

    expect_eq_i(static_cast<long>(r.sequence.size()), N, "one token per residue");
    expect_eq_i(static_cast<long>(r.tokens.size()), N, "token count");
    for (int i = 0; i < N; ++i) {
        expect_true(r.tokens[i] >= 0 && r.tokens[i] < 20, "token in range");
        expect_true(r.sequence[i] >= 'A' && r.sequence[i] <= 'Z', "valid AA letter");
    }
    std::printf("    sampled sequence: %s\n", r.sequence.c_str());
    std::remove(path.c_str());
}

int main() {
    std::printf("== test_ifold ==\n");
    return boltztest::run_all();
}
