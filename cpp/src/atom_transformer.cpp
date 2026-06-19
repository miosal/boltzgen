#include "boltz/atom_transformer.hpp"

#include "boltz/atom_windowing.hpp"

namespace boltz {

std::vector<float> windowed_atom_attention(const std::vector<float>& q,
                                           const std::vector<float>& kv, int N, int d, int W,
                                           int H, const AttnPairBiasWeights& attn,
                                           const std::vector<float>& per_head_bias) {
    const int K = N / W;
    const int nH = attn.num_heads;

    // Gather the H-key block for every window from the kv source.
    std::vector<float> keys = single_to_keys(kv, N, d, W, H);  // [K*H*d]
    // Key validity: gather a ones vector -> 0 where boundary-filled.
    std::vector<float> ones(N, 1.0f);
    std::vector<float> key_valid = single_to_keys(ones, N, 1, W, H);  // [K*H]

    std::vector<float> out((size_t)N * d);
    for (int k = 0; k < K; ++k) {
        // Window queries [W*d].
        std::vector<float> qwin(q.begin() + (size_t)k * W * d, q.begin() + (size_t)(k + 1) * W * d);
        // Window keys [H*d].
        std::vector<float> kwin(keys.begin() + (size_t)k * H * d, keys.begin() + (size_t)(k + 1) * H * d);
        // Per-head bias for this window [W*H*nH] (zeros if none supplied).
        std::vector<float> bias((size_t)W * H * nH, 0.0f);
        if (!per_head_bias.empty()) {
            const size_t off = (size_t)k * W * H * nH;
            for (size_t i = 0; i < bias.size(); ++i) bias[i] = per_head_bias[off + i];
        }
        // Mask [W*H]: keys valid per the gather (broadcast across queries).
        std::vector<float> mask((size_t)W * H);
        for (int wq = 0; wq < W; ++wq)
            for (int hk = 0; hk < H; ++hk) mask[wq * H + hk] = key_valid[k * H + hk];

        auto owin = attention_pair_bias_cross(qwin, kwin, bias, mask, W, H, attn);
        for (int i = 0; i < W * d; ++i) out[(size_t)k * W * d + i] = owin[i];
    }
    return out;
}

}  // namespace boltz
