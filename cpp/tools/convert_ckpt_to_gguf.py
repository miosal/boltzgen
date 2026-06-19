"""Convert a BoltzGen PyTorch checkpoint (.ckpt) to .gguf for boltz.cpp.

Reads a PyTorch-Lightning checkpoint, extracts the model weights, and writes
each floating-point tensor to a GGUF file (fp32) under its state_dict key, using
the dependency-free GGUFWriter in this directory. The C++ engine's graph builder
maps these names to its own tensors, so this converter keeps the original names
(verbatim) and does not rebuild the model.

Tensor layout: a torch weight of shape [d0, d1, ..., dk] is stored with ggml
dims reversed ([dk, ..., d0]) so that ne0 (the fastest ggml axis) is the last,
contiguous torch axis. A standard row-major flatten then matches ggml's layout
(e.g. nn.Linear [out, in] -> ggml ne=[in, out]).

Requires `torch` (only to read the checkpoint). Not exercised by the C++ test
suite, which validates the *output format* via the stdlib round-trip test
(test_python_gguf); run this on a machine that has the checkpoint + torch.

Usage:
  python3 convert_ckpt_to_gguf.py in.ckpt out.gguf [--arch boltzgen-ifold] [--ema]
"""

import argparse
import sys

from gguf_writer import GGUFWriter


def _extract_state_dict(ckpt, use_ema):
    # Lightning stores weights under "state_dict"; a raw dict is also accepted.
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if use_ema:
        # EMA weights (when present) are stored alongside the live weights with
        # an "ema"/"ema_model"/"_ema" marker; prefer them and strip the prefix.
        ema = {}
        for k, v in sd.items():
            for marker in ("ema_model.", "ema.", "_ema."):
                if k.startswith(marker):
                    ema[k[len(marker):]] = v
                    break
        if ema:
            print(f"using {len(ema)} EMA tensors")
            return ema
        print("warning: --ema set but no EMA tensors found; using live weights",
              file=sys.stderr)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--arch", default="boltzgen")
    ap.add_argument("--ema", action="store_true")
    args = ap.parse_args()

    import torch  # imported lazily so format-only usage needs no torch

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ckpt, args.ema)

    w = GGUFWriter()
    w.add_str("general.architecture", args.arch)

    n_written = 0
    n_skipped = 0
    for name, t in sd.items():
        if not torch.is_tensor(t) or not torch.is_floating_point(t):
            n_skipped += 1
            continue
        t = t.detach().to(torch.float32).contiguous()
        ne_dims = list(reversed(list(t.shape))) if t.dim() > 0 else [1]
        w.add_tensor(name, ne_dims, t.view(-1).tolist())
        n_written += 1

    w.write(args.out)
    print(f"wrote {n_written} tensors to {args.out} (skipped {n_skipped} non-float)")


if __name__ == "__main__":
    main()
