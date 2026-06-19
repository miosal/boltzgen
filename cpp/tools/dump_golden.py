"""Dump golden reference tensors from the PyTorch BoltzGen model for C++ parity.

For a fixed seed and a fixed input structure, run a reference model forward and
write its input features and outputs as .npy fixtures (via the stdlib npy_writer)
that the C++ parity harness (test_parity / boltz::load_npy + compare) checks
against. This is the PyTorch side of numeric-parity validation.

Requires `torch` + the `boltzgen` package + the trained checkpoint — i.e. run it
where the weights live (the build container blocks HuggingFace). The C++ side
(npy reader + compare) is already tested cross-language; committing the .npy
fixtures this produces closes numeric parity in CI.

Usage:
  python3 dump_golden.py <ckpt> <design_spec.yaml> <out_dir> [--seed 0]
"""

import argparse
import os

from npy_writer import write_npy


def _flatten(t):
    return [float(x) for x in t.detach().to("cpu").reshape(-1).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("spec")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch  # lazy: format-only usage of npy_writer needs no torch

    torch.manual_seed(args.seed)

    # Build the model + a single featurized input from the design spec. The exact
    # construction mirrors boltzgen.task.predict.predict; kept here as the
    # reference entry point so dumps match what the C++ engine reproduces.
    from boltzgen.model.models.boltz import Boltz  # noqa: F401  (import gates on env)

    raise SystemExit(
        "dump_golden.py is the reference scaffold. Wire it to the predict-path "
        "featurizer + Boltz.load_from_checkpoint on a machine with the weights, "
        "then for each tensor of interest call:\n"
        "    write_npy(os.path.join(out_dir, name + '.npy'), tuple(t.shape), _flatten(t))\n"
        "Suggested dumps: input features (res_type, coords, token masks), trunk "
        "s/z after each recycle, distogram logits, and (for inverse fold) the "
        "per-position sequence logits. The C++ parity test then loads these and "
        "asserts compare(got, golden).max_abs < tol."
    )


if __name__ == "__main__":
    main()
