"""Write a small known GGUF file using the hand-rolled writer.

Used by the C++ cross-language round-trip test (test_python_gguf) to confirm the
Python writer's output is byte-compatible with the C++ WeightStore (ggml loader).
Stdlib only — runnable without numpy/torch.

Usage: python3 gen_sample_gguf.py <out_path>
"""

import sys

from gguf_writer import GGUFWriter


def main(out_path):
    w = GGUFWriter()
    w.add_str("general.architecture", "boltzgen-ifold")
    w.add_u32("boltzgen.node_dim", 128)
    # [in=3, out=2] linear weight, row-major (ne order [in, out]).
    w.add_tensor("layer.weight", [3, 2], [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    w.add_tensor("layer.bias", [2], [-1.0, 2.0])
    w.write(out_path)


if __name__ == "__main__":
    main(sys.argv[1])
