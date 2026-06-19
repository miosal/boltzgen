// boltzcpp_design: a design-style pipeline DEMONSTRATION that assembles the real
// featurizers end to end:
//
//   parse CIF -> token_features + single-sequence MSA + atom topology
//     -> InputEmbedder (s init) + RelativePositionEncoder (z init)
//     -> Pairformer trunk
//     -> diffusion score-net + reverse-diffusion sampler -> coords
//     -> mmCIF out
//
// Unlike fold_demo's ad-hoc embedding, this drives the actual featurization
// components (token_features, single_sequence_msa_features, token_input_embedding,
// relative_position_encode). Weights are synthetic, so output is not meaningful;
// this demonstrates the DESIGN pipeline assembled and executing on a real
// structure. Real weights + arbitrary-ligand chemistry remain network-gated.
#include "boltz/cif.hpp"
#include "boltz/const.hpp"
#include "boltz/atom_featurizer.hpp"
#include "boltz/demo_synth.hpp"
#include "boltz/diffusion_blocks.hpp"
#include "boltz/diffusion_sampler.hpp"
#include "boltz/diffusion_schedule.hpp"
#include "boltz/diffusion_transformer.hpp"
#include "boltz/featurizer.hpp"
#include "boltz/input_embedder.hpp"
#include "boltz/mmcif_writer.hpp"
#include "boltz/rel_pos.hpp"
#include "boltz/trunk.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace boltz;

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

    const int ts = 16, tz = 16, sH = 2, tH = 2, thd = 4, num_blocks = 2;
    const int ddim = 16, dheads = 2, dlayers = 2;

    // --- featurization (real components) ---
    std::vector<std::string> comps(N);
    std::vector<int> seqs(N);
    std::vector<float> coords(N * 3);
    for (int i = 0; i < N; ++i) {
        comps[i] = residues[i].comp_id; seqs[i] = residues[i].seq_id;
        coords[i * 3] = residues[i].ca_x; coords[i * 3 + 1] = residues[i].ca_y; coords[i * 3 + 2] = residues[i].ca_z;
    }
    auto tf = token_features(comps, seqs);
    auto msa = single_sequence_msa_features(tf.res_type, tf.n, tf.num_token_types);
    auto af = atom_features(comps);
    std::vector<float> deletion_mean(N, 0.0f);

    // --- InputEmbedder (token-level s init) ---
    const int T = tf.num_token_types;
    InputEmbedderWeights iew; iew.token_s = ts; iew.num_tokens = T;
    iew.res_type_w = demo::fv(ts * T, 1.0f);
    iew.profile_w = demo::fv(ts * (T + 1), 2.0f);
    iew.add_mol_type = true; iew.mol_type_emb = demo::fv(4 * ts, 3.0f);
    auto s = token_input_embedding(iew, N, tf.res_type, msa.profile, deletion_mean, tf.mol_type, {}, {});

    // --- RelativePositionEncoder (pair z init) ---
    RelPosInputs rp;
    rp.feature_asym_id.assign(N, 0); rp.entity_id.assign(N, 0); rp.sym_id.assign(N, 0);
    rp.feature_residue_index = seqs; rp.token_index = tf.token_index;
    auto z = relative_position_encode(rp, N, tz, demo::fv(tz * rel_pos_feature_dim(32, 2), 4.0f));

    // --- trunk ---
    std::vector<PairformerWeights> blocks;
    for (int b = 0; b < num_blocks; ++b) blocks.push_back(demo::mk_block(ts, tz, sH, tH, thd, 100.0f * (b + 1)));
    std::vector<float> tmask(N, 1.0f), pmask(N * N, 1.0f);
    auto trunk = pairformer_trunk(s, z, tmask, pmask, N, blocks);

    // --- diffusion refinement conditioned on trunk s ---
    DiffusionSchedule sched(DiffusionScheduleConfig{});
    auto sigmas = sched.sample_schedule_af3(12);
    auto atom_embed = demo::fv(ddim * 3, 8.0f), atom_decode = demo::fv(3 * ddim, 9.0f);
    auto four_w = demo::fv(ddim, 10.0f), four_b = demo::fv(ddim, 11.0f);
    std::vector<DiffusionTransformerLayerWeights> dlw;
    for (int l = 0; l < dlayers; ++l) dlw.push_back(demo::mk_difflayer(ddim, ts, dheads, 200.0f * (l + 1)));
    std::vector<std::vector<float>> dbiases(dlayers, std::vector<float>(N * N * dheads, 0.0f));

    auto net = [&](const std::vector<float>& scaled, double t) {
        auto te = fourier_embedding(static_cast<float>(t), four_w, four_b);
        std::vector<float> a(N * ddim);
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < ddim; ++d) { float v = te[d]; for (int c = 0; c < 3; ++c) v += atom_embed[d * 3 + c] * scaled[i * 3 + c]; a[i * ddim + d] = v; }
        auto ao = diffusion_transformer(a, trunk.s, dbiases, tmask, N, dlw);
        std::vector<float> r(N * 3);
        for (int i = 0; i < N; ++i) for (int c = 0; c < 3; ++c) { float v = 0; for (int d = 0; d < ddim; ++d) v += atom_decode[c * ddim + d] * ao[i * ddim + d]; r[i * 3 + c] = v; }
        return r;
    };
    auto denoise = [&](const std::vector<float>& c, double t_hat) { return preconditioned_forward(c, t_hat, sched, net); };
    unsigned st = 7;
    auto randn = [&]() { st = st * 1664525u + 1013904223u; return static_cast<float>(st >> 9) / 8388608.0f - 1.0f; };
    auto designed = sample_diffusion(sigmas, N, tmask, denoise, randn, SamplerParams{});

    std::printf("structure  : %s chain %s (%d residues, %d atoms)\n", cif.c_str(), chain.c_str(), N, af.n_atoms);
    std::printf("featurizer : token_features + single-seq MSA + atom topology\n");
    std::printf("embed      : InputEmbedder s[%d,%d] + RelPos z[%d,%d,%d]\n", N, ts, N, N, tz);
    std::printf("trunk      : %d Pairformer blocks\n", num_blocks);
    std::printf("diffusion  : %d steps -> %d designed coords\n", (int)sigmas.size() - 1, N);

    if (!out_path.empty()) {
        std::vector<CifAtom> out_atoms;
        for (int i = 0; i < N; ++i) {
            CifAtom a; a.group = "ATOM"; a.atom_id = "CA"; a.comp_id = comps[i];
            a.asym_id = chain; a.seq_id = seqs[i];
            a.x = designed[i * 3]; a.y = designed[i * 3 + 1]; a.z = designed[i * 3 + 2];
            out_atoms.push_back(a);
        }
        write_cif_file(out_path, out_atoms, "boltzcpp_design");
        std::printf("output     : %s\n", out_path.c_str());
    }
    return 0;
}
