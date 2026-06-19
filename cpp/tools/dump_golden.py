"""Dump golden reference tensors from the PyTorch BoltzGen inverse-fold model.

For a fixed seed and a fixed (controlled) input scenario, run the *real* trained
inverse-folding encoder + decoder and serialize their inputs and outputs as .npy
fixtures (via the stdlib npy_writer) that the C++ parity harness (test_parity /
boltz::load_npy + compare) checks against. This is the PyTorch side of numeric
parity validation on trained weights.

Scope (be honest about what this validates):
  * It exercises `inverse_folding_encoder.*` (linear_token_to_node,
    linear_token_to_pair, 3x MLPAttnGNN) and `structure_module.*`
    (seq_to_s, 3x MLPAttnGNNDecoder, predictor) -- i.e. the GNN core that is
    314 of the checkpoint's 357 tensors -- with the *real trained weights*.
  * The encoder's node features `s_inputs` are normally produced upstream by the
    InputEmbedder (43 tensors, a large atom-attention encoder) and the edge
    features by the hand-rolled featurizer on a real structure. Reproducing those
    in C++ with trained weights is a separate, larger port (and the featurizer
    needs HF-gated mol data); they are NOT covered here. Instead we feed
    fixed-seed synthetic-but-validly-shaped `s_inputs` / edge features so the
    encoder+decoder numerics can be compared exactly: identical inputs + identical
    trained weights => the C++ and PyTorch outputs must match to fp32 tolerance.
  * The autoregressive sampler (`InverseFoldingDecoder.sample`) is stochastic
    (randperm + multinomial); here we validate the deterministic decoder
    `forward()` logits given a dumped (post-RNG) sequence-visibility input.

Requires `torch` + the boltzgen source on sys.path + the trained checkpoint. The
checkpoint's `hyper_parameters` reference omegaconf/boltzgen config classes that
need not be importable: we instantiate the modules directly and load only the
relevant state_dict subset.

Usage:
  python3 dump_golden.py <ckpt> <out_dir> [--seed 0] [--n 24] [--k 12]
"""

import argparse
import os
import pickle
import sys

from npy_writer import write_npy


def _safe_torch_load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        pass

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            if isinstance(s, dict):
                self.__dict__.update(s)

        def __reduce__(self):
            return (_Stub, ())

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError, ImportError):
                return type(name, (_Stub,), {})

    class _pickle_mod:
        Unpickler = _SafeUnpickler
        load = staticmethod(pickle.load)

    return torch.load(path, map_location="cpu", weights_only=False,
                      pickle_module=_pickle_mod)


def _subdict(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _dump(out_dir, name, t):
    write_npy(os.path.join(out_dir, name + ".npy"), tuple(t.shape),
              [float(x) for x in t.detach().to("cpu").reshape(-1).tolist()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=24, help="number of nodes/residues")
    ap.add_argument("--k", type=int, default=12, help="kNN neighbours per node")
    ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__), "..", ".."),
                    help="repo root (for boltzgen on sys.path)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(os.path.abspath(args.src), "src"))
    import torch
    from boltzgen.model.modules.inverse_fold import (
        InverseFoldingEncoder,
        InverseFoldingDecoder,
    )

    sd = _safe_torch_load(args.ckpt)
    sd = sd.get("state_dict", sd)

    # ---- infer dims from the trained tensors (no config needed) -------------
    node_dim = sd["inverse_folding_encoder.linear_token_to_node.weight"].shape[0]   # 128
    token_s = sd["inverse_folding_encoder.linear_token_to_node.weight"].shape[1]    # 384
    pair_dim = sd["inverse_folding_encoder.linear_token_to_pair.weight"].shape[0]   # 128
    edge_in = sd["inverse_folding_encoder.linear_token_to_pair.weight"].shape[1]    # 332
    num_heads = sd["inverse_folding_encoder.encoder_layers.0.attn_weight_mlp.4.weight"].shape[0]
    n_enc = len({k.split(".")[2] for k in sd if k.startswith("inverse_folding_encoder.encoder_layers.")})
    n_dec = len({k.split(".")[2] for k in sd if k.startswith("structure_module.decoder_layers.")})
    num_tokens = sd["structure_module.predictor.weight"].shape[0]                   # 33
    print(f"dims: node={node_dim} pair={pair_dim} token_s={token_s} edge_in={edge_in} "
          f"heads={num_heads} enc_layers={n_enc} dec_layers={n_dec} num_tokens={num_tokens}")
    print(f"predictor.weight norm = {sd['structure_module.predictor.weight'].norm().item():.4f} "
          "(non-zero => trained, not the zero-init)")

    # ---- build + load the real modules --------------------------------------
    enc = InverseFoldingEncoder(
        atom_s=128, atom_z=16, token_s=token_s, token_z=128,
        node_dim=node_dim, pair_dim=pair_dim, hidden_dim=node_dim,
        dropout=0.0, softmax_dropout=0.0, num_encoder_layers=n_enc,
        num_heads=num_heads, topk=args.k, enable_input_embedder=True)
    miss = enc.load_state_dict(_subdict(sd, "inverse_folding_encoder."), strict=True)
    enc.eval()

    dec = InverseFoldingDecoder(
        atom_s=128, atom_z=16, token_s=token_s, token_z=128,
        node_dim=node_dim, pair_dim=pair_dim, hidden_dim=node_dim,
        dropout=0.0, softmax_dropout=0.0, num_encoder_layers=n_enc,
        num_decoder_layers=n_dec, num_heads=num_heads, topk=args.k,
        inverse_fold_restriction=[], sampling_temperature=0.1)
    dec.load_state_dict(_subdict(sd, "structure_module."), strict=True)
    dec.eval()

    # ---- a fixed-seed controlled scenario -----------------------------------
    torch.manual_seed(args.seed)
    N, K = args.n, min(args.k, args.n)

    # kNN graph from random 3-D points (self included, like init_knn_graph).
    coords = torch.randn(N, 3)
    dists = torch.cdist(coords[None], coords[None])[0]
    src_idx = torch.topk(dists, K, largest=False).indices               # [N, K]
    dst_idx = torch.arange(N)[:, None].expand(-1, K)                    # [N, K]
    src_idx = src_idx.reshape(-1).contiguous()
    dst_idx = dst_idx.reshape(-1).contiguous()
    edge_idx = torch.stack([src_idx, dst_idx], dim=0)
    E = src_idx.numel()

    s_inputs = torch.randn(N, token_s)
    pair_input = torch.randn(E, edge_in)

    with torch.no_grad():
        # encoder
        s = enc.linear_token_to_node(s_inputs)
        z = enc.linear_token_to_pair(pair_input)
        for layer in enc.encoder_layers:
            s, z = layer(s, z, edge_idx)
        s_enc, z_enc = s, z

        # decoder forward: build the (post-RNG) per-edge sequence visibility input,
        # exactly as InverseFoldingDecoder.forward does, then dump it so C++ can
        # reproduce seq_to_s + the decoder GNN + predictor deterministically.
        rand = torch.rand(N)
        vis = (rand[src_idx] < rand[dst_idx])                          # [E] bool
        res_ids = torch.randint(0, num_tokens, (N,))
        res_type_clone = torch.nn.functional.one_hot(res_ids, num_tokens).float()
        res_type_vis = res_type_clone[src_idx] * vis[:, None].float()  # [E, num_tokens]

        res_rep = dec.seq_to_s(res_type_vis)
        neighbors_rep = torch.cat([z_enc, s_enc[src_idx] + res_rep], dim=-1)
        sd_ = s_enc.clone()
        for layer in dec.decoder_layers:
            sd_ = layer(sd_, neighbors_rep, edge_idx)
        logits = dec.predictor(sd_)                                    # [N, num_tokens]

    # ---- write fixtures ------------------------------------------------------
    _dump(args.out_dir, "edge_src", src_idx.float())
    _dump(args.out_dir, "edge_dst", dst_idx.float())
    _dump(args.out_dir, "s_inputs", s_inputs)
    _dump(args.out_dir, "pair_input", pair_input)
    _dump(args.out_dir, "res_type_vis", res_type_vis)
    _dump(args.out_dir, "s_enc", s_enc)
    _dump(args.out_dir, "z_enc", z_enc)
    _dump(args.out_dir, "logits", logits)

    with open(os.path.join(args.out_dir, "meta.txt"), "w") as f:
        f.write(f"N={N}\nK={K}\nE={E}\nnode_dim={node_dim}\npair_dim={pair_dim}\n"
                f"token_s={token_s}\nedge_in={edge_in}\nnum_heads={num_heads}\n"
                f"n_enc={n_enc}\nn_dec={n_dec}\nnum_tokens={num_tokens}\nseed={args.seed}\n")

    print(f"wrote golden fixtures to {args.out_dir} (N={N}, E={E})")
    print(f"  s_enc:  mean={s_enc.mean():.4f} std={s_enc.std():.4f}")
    print(f"  z_enc:  mean={z_enc.mean():.4f} std={z_enc.std():.4f}")
    print(f"  logits: mean={logits.mean():.4f} std={logits.std():.4f} "
          f"argmax(first 8)={logits[:8].argmax(-1).tolist()}")


if __name__ == "__main__":
    main()
