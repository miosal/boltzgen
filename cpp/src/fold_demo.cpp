// boltzcpp_fold: a folding-style structure-prediction DEMONSTRATION that wires
// the validated components end to end on a real protein CIF:
//
//   parse CIF -> CA coords + residue types
//     -> embed s, gaussian-smear pair z
//     -> Pairformer trunk (s, z)
//     -> distogram head + confidence (pLDDT) head
//     -> diffusion score-net (atom embed + Fourier time + DiffusionTransformer
//        conditioned on s + atom decode) driven by the reverse-diffusion sampler
//     -> refined coordinates -> write mmCIF (pLDDT in B-factor)
//
// This exercises the trunk + diffusion + confidence subsystems together and
// RUNS on example structures. Weights are synthetic (deterministic), so output
// is not meaningful — it demonstrates the assembled pipeline executing. Real
// weights + the atom featurizer are the remaining integration step.
#include "boltz/affinity.hpp"
#include "boltz/attention_pair_bias.hpp"
#include "boltz/cif.hpp"
#include "boltz/confidence.hpp"
#include "boltz/const.hpp"
#include "boltz/diffusion_blocks.hpp"
#include "boltz/diffusion_sampler.hpp"
#include "boltz/diffusion_schedule.hpp"
#include "boltz/diffusion_transformer.hpp"
#include "boltz/featurizer.hpp"
#include "boltz/mmcif_writer.hpp"
#include "boltz/trunk.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace boltz;

namespace {

// Deterministic synthetic weight fill.
std::vector<float> fv(int n, float seed) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = 0.05f * std::sin(seed + 0.137f * i);
    return v;
}
std::vector<float> ones(int n) { return std::vector<float>(n, 1.0f); }
std::vector<float> zeros(int n) { return std::vector<float>(n, 0.0f); }

TriMulWeights mk_trimul(int D, float s) {
    TriMulWeights w; w.dim = D;
    w.norm_in_w = ones(D); w.norm_in_b = zeros(D); w.norm_out_w = ones(D); w.norm_out_b = zeros(D);
    w.p_in = fv(2 * D * D, s + 1); w.g_in = fv(2 * D * D, s + 2);
    w.p_out = fv(D * D, s + 3); w.g_out = fv(D * D, s + 4);
    return w;
}
TriAttnWeights mk_triattn(int C, int H, int hd, float s) {
    TriAttnWeights w; w.c_in = C; w.num_heads = H; w.head_dim = hd; w.inf = 1e9f;
    w.norm_w = ones(C); w.norm_b = zeros(C);
    w.tri_w = fv(H * C, s + 1); w.q_w = fv(H * hd * C, s + 2); w.k_w = fv(H * hd * C, s + 3);
    w.v_w = fv(H * hd * C, s + 4); w.g_w = fv(H * hd * C, s + 5); w.o_w = fv(C * H * hd, s + 6);
    return w;
}
TransitionWeights mk_trans(int dim, int hid, float s) {
    TransitionWeights t; t.dim = dim; t.hidden = hid; t.out = dim;
    t.norm_w = ones(dim); t.norm_b = zeros(dim);
    t.fc1 = fv(hid * dim, s + 1); t.fc2 = fv(hid * dim, s + 2); t.fc3 = fv(dim * hid, s + 3);
    return t;
}
AttnPairBiasWeights mk_attn(int cs, int cz, int H, float s) {
    AttnPairBiasWeights w; w.c_s = cs; w.c_z = cz; w.num_heads = H; w.inf = 1e6f;
    w.q_w = fv(cs * cs, s + 1); w.q_b = fv(cs, s + 1.5f); w.k_w = fv(cs * cs, s + 2);
    w.v_w = fv(cs * cs, s + 3); w.g_w = fv(cs * cs, s + 4);
    w.z_norm_w = ones(cz); w.z_norm_b = zeros(cz); w.z_lin_w = fv(H * cz, s + 5); w.o_w = fv(cs * cs, s + 6);
    return w;
}
PairformerWeights mk_block(int ts, int tz, int sH, int tH, int thd, float s) {
    PairformerWeights w; w.token_s = ts; w.token_z = tz;
    w.tri_mul_out = mk_trimul(tz, s + 10); w.tri_mul_in = mk_trimul(tz, s + 20);
    w.tri_att_start = mk_triattn(tz, tH, thd, s + 30); w.tri_att_end = mk_triattn(tz, tH, thd, s + 40);
    w.transition_z = mk_trans(tz, tz * 2, s + 50); w.transition_s = mk_trans(ts, ts * 2, s + 60);
    w.pre_norm_s_w = ones(ts); w.pre_norm_s_b = zeros(ts);
    w.attention = mk_attn(ts, tz, sH, s + 70);
    return w;
}
AdaLNWeights mk_adaln(int dim, int dc, float s) {
    AdaLNWeights w; w.dim = dim; w.dim_cond = dc;
    w.s_norm_w = ones(dc); w.s_scale_w = fv(dim * dc, s + 1); w.s_scale_b = fv(dim, s + 2); w.s_bias_w = fv(dim * dc, s + 3);
    return w;
}
DiffusionTransformerLayerWeights mk_difflayer(int dim, int dc, int H, float s) {
    DiffusionTransformerLayerWeights w; w.dim = dim; w.dim_cond = dc; w.num_heads = H;
    w.adaln = mk_adaln(dim, dc, s);
    w.attn.c_s = dim; w.attn.c_z = 0; w.attn.num_heads = H; w.attn.compute_pair_bias = false; w.attn.inf = 1e6f;
    w.attn.q_w = fv(dim * dim, s + 10); w.attn.q_b = fv(dim, s + 10.5f); w.attn.k_w = fv(dim * dim, s + 11);
    w.attn.v_w = fv(dim * dim, s + 12); w.attn.g_w = fv(dim * dim, s + 13); w.attn.o_w = fv(dim * dim, s + 14);
    w.op_w = fv(dim * dc, s + 15); w.op_b = fv(dim, s + 16);
    w.transition.dim = dim; w.transition.dim_cond = dc; w.transition.dim_inner = dim * 2;
    w.transition.adaln = mk_adaln(dim, dc, s + 20);
    w.transition.swish_gate_w = fv(2 * (dim * 2) * dim, s + 21);
    w.transition.a_to_b_w = fv((dim * 2) * dim, s + 22);
    w.transition.b_to_a_w = fv(dim * (dim * 2), s + 23);
    w.transition.op_w = fv(dim * dc, s + 24); w.transition.op_b = fv(dim, s + 25);
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <structure.cif> <chain> [--out path]\n", argv[0]);
        return 2;
    }
    const std::string cif = argv[1], chain = argv[2];
    std::string out_path;
    for (int i = 3; i < argc; ++i)
        if (std::string(argv[i]) == "--out" && i + 1 < argc) out_path = argv[++i];

    auto atoms = parse_cif_file(cif);
    auto residues = residues_with_ca(atoms, chain);
    if (residues.empty()) { std::fprintf(stderr, "no residues for chain %s\n", chain.c_str()); return 1; }
    const int N = static_cast<int>(residues.size());

    // Dimensions (small synthetic model).
    const int ts = 16, tz = 16, sH = 2, tH = 2, thd = 4, num_blocks = 2;
    const int gaussians = 16, num_embed = 21, dist_bins = 64, plddt_bins = 50;
    const int ddim = 16, dheads = 2, dlayers = 2;

    // --- features ---
    std::vector<float> coords(N * 3);
    for (int i = 0; i < N; ++i) { coords[i * 3] = residues[i].ca_x; coords[i * 3 + 1] = residues[i].ca_y; coords[i * 3 + 2] = residues[i].ca_z; }

    auto tok_embed = fv(num_embed * ts, 1.0f);
    std::vector<float> s(N * ts);
    for (int i = 0; i < N; ++i) {
        int tok = canonical_index(residues[i].comp_id);
        if (tok < 0) tok = num_embed - 1;
        for (int d = 0; d < ts; ++d) s[i * ts + d] = tok_embed[tok * ts + d];
    }
    auto edge_w = fv(tz * gaussians, 2.0f);
    std::vector<float> z(N * N * tz);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float dx = coords[i * 3] - coords[j * 3], dy = coords[i * 3 + 1] - coords[j * 3 + 1], dz = coords[i * 3 + 2] - coords[j * 3 + 2];
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            Matrix sm = gaussian_smearing({dist}, 0.0f, 22.0f, gaussians);
            for (int c = 0; c < tz; ++c) { float v = 0; for (int g = 0; g < gaussians; ++g) v += edge_w[c * gaussians + g] * sm.data[g]; z[(i * N + j) * tz + c] = v; }
        }

    // --- trunk ---
    std::vector<PairformerWeights> blocks;
    for (int b = 0; b < num_blocks; ++b) blocks.push_back(mk_block(ts, tz, sH, tH, thd, 100.0f * (b + 1)));
    std::vector<float> tmask(N, 1.0f), pmask(N * N, 1.0f);
    auto trunk = pairformer_trunk(s, z, tmask, pmask, N, blocks);

    // --- distogram + confidence ---
    auto disto = distogram_head(trunk.z, N, tz, dist_bins, fv(dist_bins * tz, 3.0f), fv(dist_bins, 3.5f));
    ConfidenceWeights cw; cw.token_s = ts; cw.token_z = tz; cw.num_pae_bins = 64; cw.num_pde_bins = 64; cw.num_plddt_bins = plddt_bins;
    cw.to_pae = fv(64 * tz, 4); cw.to_pde = fv(64 * tz, 5); cw.to_plddt = fv(plddt_bins * ts, 6); cw.to_resolved = fv(2 * ts, 7);
    auto conf = confidence_heads(trunk.s, trunk.z, N, cw);
    auto plddt = aggregated_metric(conf.plddt_logits, N, plddt_bins, 1.0f);
    float mean_plddt = 0; for (float v : plddt) mean_plddt += v; mean_plddt /= N;

    // --- diffusion refinement (residues as atoms, conditioned on trunk s) ---
    DiffusionSchedule sched(DiffusionScheduleConfig{});
    auto sigmas = sched.sample_schedule_af3(12);
    auto atom_embed = fv(ddim * 3, 8.0f), atom_decode = fv(3 * ddim, 9.0f);
    auto four_w = fv(ddim, 10.0f), four_b = fv(ddim, 11.0f);
    std::vector<DiffusionTransformerLayerWeights> dlayers_w;
    for (int l = 0; l < dlayers; ++l) dlayers_w.push_back(mk_difflayer(ddim, ts, dheads, 200.0f * (l + 1)));
    std::vector<std::vector<float>> dbiases(dlayers, std::vector<float>(N * N * dheads, 0.0f));

    auto net = [&](const std::vector<float>& scaled, double t) {
        auto te = fourier_embedding(static_cast<float>(t), four_w, four_b);
        std::vector<float> a(N * ddim);
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < ddim; ++d) { float v = te[d]; for (int c = 0; c < 3; ++c) v += atom_embed[d * 3 + c] * scaled[i * 3 + c]; a[i * ddim + d] = v; }
        auto a_out = diffusion_transformer(a, trunk.s, dbiases, tmask, N, dlayers_w);
        std::vector<float> r(N * 3);
        for (int i = 0; i < N; ++i) for (int c = 0; c < 3; ++c) { float v = 0; for (int d = 0; d < ddim; ++d) v += atom_decode[c * ddim + d] * a_out[i * ddim + d]; r[i * 3 + c] = v; }
        return r;
    };
    auto denoise = [&](const std::vector<float>& c, double t_hat) { return preconditioned_forward(c, t_hat, sched, net); };
    unsigned st = 2024;
    auto randn = [&]() { st = st * 1664525u + 1013904223u; return static_cast<float>(st >> 9) / 8388608.0f - 1.0f; };
    auto refined = sample_diffusion(sigmas, N, tmask, denoise, randn, SamplerParams{});

    // --- affinity head (designate the last 8 residues as a pseudo-ligand) ---
    auto mk_linb = [](int in, int out, float s) {
        LinearB L; L.in = in; L.out = out; L.w = fv(in * out, s); L.b = fv(out, s + 0.5f); return L;
    };
    AffinityWeights aw; aw.token_z = tz; aw.token_s = ts;
    aw.out_l1 = mk_linb(tz, tz, 30); aw.out_l2 = mk_linb(tz, ts, 31);
    aw.val_l1 = mk_linb(ts, ts, 32); aw.val_l2 = mk_linb(ts, ts, 33); aw.val_l3 = mk_linb(ts, 1, 34);
    aw.sco_l1 = mk_linb(ts, ts, 35); aw.sco_l2 = mk_linb(ts, ts, 36); aw.sco_l3 = mk_linb(ts, 1, 37);
    aw.binary = mk_linb(1, 1, 38);
    std::vector<float> lig_mask(N, 0.0f), rec_mask(N, 0.0f);
    for (int i = 0; i < N; ++i) (i >= N - 8 ? lig_mask : rec_mask)[i] = 1.0f;
    auto aff = affinity_head(trunk.z, lig_mask, rec_mask, N, aw);

    std::printf("structure : %s  chain %s (%d residues)\n", cif.c_str(), chain.c_str(), N);
    std::printf("trunk     : token_s=%d token_z=%d blocks=%d\n", ts, tz, num_blocks);
    std::printf("distogram : %d x %d x %d bins\n", N, N, dist_bins);
    std::printf("mean pLDDT: %.3f (synthetic weights)\n", mean_plddt);
    std::printf("diffusion : %d steps, refined %d atom coords\n", (int)sigmas.size() - 1, N);
    std::printf("affinity  : value=%.3f score=%.3f binary_logit=%.3f (pseudo-ligand, synthetic)\n",
                aff.pred_value, aff.pred_score, aff.logits_binary);

    if (!out_path.empty()) {
        std::vector<CifAtom> out_atoms;
        for (int i = 0; i < N; ++i) {
            CifAtom a; a.group = "ATOM"; a.atom_id = "CA"; a.comp_id = residues[i].comp_id;
            a.asym_id = chain; a.seq_id = residues[i].seq_id;
            a.x = refined[i * 3]; a.y = refined[i * 3 + 1]; a.z = refined[i * 3 + 2];
            out_atoms.push_back(a);
        }
        write_cif_file(out_path, out_atoms, "boltzcpp_fold");
        std::printf("output    : %s\n", out_path.c_str());
    }
    return 0;
}
