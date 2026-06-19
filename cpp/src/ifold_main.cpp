// boltzcpp_ifold: run the inverse-folding pipeline on a structure file.
//
//   boltzcpp_ifold <model.gguf> <structure.cif> <chain> [designStart..designEnd]
//
// Parses the mmCIF, builds per-residue CA coordinates for the chosen chain, runs
// the ggml inverse-folding model, and prints the sampled sequence. With the
// converted checkpoint this produces designed sequences; with a synthetic gguf
// it demonstrates the pipeline executing end to end.
#include "boltz/cif.hpp"
#include "boltz/const.hpp"
#include "boltz/ifold_model.hpp"
#include "boltz/mmcif_writer.hpp"
#include "boltz/residue_constraints.hpp"
#include "boltz/weight_store.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <structure.cif> <chain> "
                     "[designStart..designEnd]\n",
                     argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string cif_path = argv[2];
    const std::string chain = argv[3];

    auto atoms = boltz::parse_cif_file(cif_path);
    auto residues = boltz::residues_with_ca(atoms, chain);
    if (residues.empty()) {
        std::fprintf(stderr, "no residues found for chain '%s'\n", chain.c_str());
        return 1;
    }
    const int N = static_cast<int>(residues.size());

    std::vector<char> design_mask(N, 0);
    std::string design_desc = "(none)";
    std::string out_path;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            // 1-indexed positions into the chain's ordered residue list.
            auto pos = boltz::parse_range(argv[i], 0, N);
            for (int p : pos)
                if (p >= 0 && p < N) design_mask[p] = 1;
            design_desc = argv[i];
        }
    }

    boltz::WeightStore ws(gguf_path);
    boltz::IFoldModel model = boltz::load_ifold_model(ws);
    boltz::IFoldResult r = boltz::run_ifold(model, residues, design_mask);

    std::printf("structure : %s\n", cif_path.c_str());
    std::printf("chain     : %s  (%d residues)\n", chain.c_str(), N);
    std::printf("model     : node=%d layers=%d heads=%d topk=%d\n",
                model.cfg.node_dim, model.cfg.num_layers, model.cfg.num_heads,
                model.cfg.topk);
    std::printf("designed  : %s\n", design_desc.c_str());
    std::printf("sequence  : %s\n", r.sequence.c_str());

    if (!out_path.empty()) {
        // Write the backbone CA atoms with predicted residue names.
        std::vector<boltz::CifAtom> out_atoms;
        const auto& canon = boltz::canonical_tokens();
        for (int i = 0; i < N; ++i) {
            boltz::CifAtom a;
            a.group = "ATOM";
            a.atom_id = "CA";
            a.comp_id = canon[r.tokens[i]];
            a.asym_id = chain;
            a.seq_id = residues[i].seq_id;
            a.x = residues[i].ca_x; a.y = residues[i].ca_y; a.z = residues[i].ca_z;
            out_atoms.push_back(a);
        }
        boltz::write_cif_file(out_path, out_atoms, "boltzcpp_ifold");
        std::printf("output    : %s\n", out_path.c_str());
    }
    return 0;
}
