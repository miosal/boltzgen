"""Write a synthetic inverse-folding gguf so the C++ pipeline can be run on real
structures before the trained checkpoint is converted.

Tensor names + metadata match boltz::load_ifold_model. Weights are small
deterministic pseudo-random values (BatchNorm/affine set to identity), so the
output is a valid run of the pipeline, not a meaningful design. Stdlib only.

Usage: python3 gen_synth_ifold_gguf.py <out.gguf>
"""

import math
import sys

from gguf_writer import GGUFWriter

NODE = 32
PAIR = 32
HID = 32
HEADS = 4
LAYERS = 6
GAUSS = 16
TOPK = 30
NEMBED = 21


def vals(name, n):
    if name.endswith(".scale") or name == "head_norm.weight":
        return [1.0] * n
    if name.endswith(".shift") or name == "head_norm.bias":
        return [0.0] * n
    base = (hash(name) % 97)
    return [0.02 * math.sin((base + i) * 0.3) for i in range(n)]


def add(w, name, ne):
    n = 1
    for d in ne:
        n *= d
    w.add_tensor(name, ne, vals(name, n))


def add_layer(w, p):
    add(w, p + "edge_l1.weight", [2 * NODE + PAIR, HID])
    add(w, p + "edge_l1.bias", [HID])
    add(w, p + "edge_l2.weight", [HID, PAIR])
    add(w, p + "edge_l2.bias", [PAIR])
    add(w, p + "edge_bn.scale", [PAIR])
    add(w, p + "edge_bn.shift", [PAIR])
    add(w, p + "aw_l1.weight", [2 * NODE + PAIR, HID])
    add(w, p + "aw_l1.bias", [HID])
    add(w, p + "aw_l2.weight", [HID, HID])
    add(w, p + "aw_l2.bias", [HID])
    add(w, p + "aw_l3.weight", [HID, HEADS])
    add(w, p + "aw_l3.bias", [HEADS])
    add(w, p + "av_l1.weight", [NODE + PAIR, HID])
    add(w, p + "av_l1.bias", [HID])
    add(w, p + "av_l2.weight", [HID, HID])
    add(w, p + "av_l2.bias", [HID])
    add(w, p + "av_l3.weight", [HID, NODE])
    add(w, p + "av_l3.bias", [NODE])
    add(w, p + "out_l.weight", [HEADS * NODE, NODE])
    add(w, p + "out_l.bias", [NODE])
    add(w, p + "out_bn.scale", [NODE])
    add(w, p + "out_bn.shift", [NODE])
    add(w, p + "ffn_l1.weight", [NODE, HID])
    add(w, p + "ffn_l1.bias", [HID])
    add(w, p + "ffn_l2.weight", [HID, NODE])
    add(w, p + "ffn_l2.bias", [NODE])
    add(w, p + "ffn_bn.scale", [NODE])
    add(w, p + "ffn_bn.shift", [NODE])


def main(out_path):
    w = GGUFWriter()
    w.add_str("general.architecture", "boltzgen-ifold")
    w.add_u32("ifold.node_dim", NODE)
    w.add_u32("ifold.pair_dim", PAIR)
    w.add_u32("ifold.hidden_dim", HID)
    w.add_u32("ifold.num_heads", HEADS)
    w.add_u32("ifold.num_layers", LAYERS)
    w.add_u32("ifold.num_gaussians", GAUSS)
    w.add_u32("ifold.topk", TOPK)
    w.add_u32("ifold.num_embed", NEMBED)

    add(w, "tok_embed.weight", [NODE, NEMBED])
    add(w, "edge_proj.weight", [GAUSS, PAIR])
    add(w, "edge_proj.bias", [PAIR])
    for l in range(LAYERS):
        add_layer(w, "enc.%d." % l)
    add(w, "head_norm.weight", [NODE])
    add(w, "head_norm.bias", [NODE])
    add(w, "head.weight", [NODE, 20])
    add(w, "head.bias", [20])
    w.write(out_path)
    print("wrote synthetic ifold model to", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
