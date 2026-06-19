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
import pickle
import sys

from gguf_writer import GGUFWriter


def _safe_torch_load(path):
    """Load a checkpoint's tensors even if non-tensor objects in it reference
    modules that aren't importable here (e.g. omegaconf / boltzgen configs in
    `hyper_parameters`). We only need the float tensors, so unresolvable classes
    are replaced by inert stand-ins while torch's own tensor-storage handling is
    left intact."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        print(f"note: {e}; retrying with a permissive unpickler", file=sys.stderr)

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

    ckpt = _safe_torch_load(args.ckpt)
    sd = _extract_state_dict(ckpt, args.ema)

    w = GGUFWriter()
    w.add_str("general.architecture", args.arch)

    # ggml caps tensor names at GGML_MAX_NAME (64). The verbatim PyTorch keys can
    # exceed that, so for the inverse-fold arch we shorten the two GNN prefixes
    # (which the C++ engine reads) and drop the upstream input_embedder.* tensors
    # (not used by the C++ inverse-fold encoder/decoder, and too deeply nested to
    # fit the name limit). The renames keep every emitted name < 64 chars.
    def transform(name: str):
        if args.arch == "boltzgen-ifold":
            if name.startswith("input_embedder."):
                return None  # unused by the C++ ifold path
            if name.startswith("inverse_folding_encoder."):
                name = "enc." + name[len("inverse_folding_encoder."):]
            elif name.startswith("structure_module."):
                name = "dec." + name[len("structure_module."):]
        return name

    n_written = 0
    n_skipped = 0
    n_dropped = 0
    too_long = []
    for name, t in sd.items():
        if not torch.is_tensor(t) or not torch.is_floating_point(t):
            n_skipped += 1
            continue
        out_name = transform(name)
        if out_name is None:
            n_dropped += 1
            continue
        if len(out_name) >= 64:
            too_long.append(out_name)
            continue
        t = t.detach().to(torch.float32).contiguous()
        ne_dims = list(reversed(list(t.shape))) if t.dim() > 0 else [1]
        w.add_tensor(out_name, ne_dims, t.view(-1).tolist())
        n_written += 1

    if too_long:
        raise SystemExit(
            f"{len(too_long)} tensor name(s) exceed ggml's 64-char limit, e.g. "
            f"'{too_long[0]}' ({len(too_long[0])} chars). Add a rename rule.")

    w.write(args.out)
    print(f"wrote {n_written} tensors to {args.out} "
          f"(skipped {n_skipped} non-float, dropped {n_dropped} unused)")


if __name__ == "__main__":
    main()
