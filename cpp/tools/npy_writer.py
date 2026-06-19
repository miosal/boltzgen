"""Stdlib-only .npy writer (NumPy format v1.0) — no numpy dependency.

Used to (a) unit-test the C++ npy reader cross-language, and (b) by a future
golden-tensor dumper so PyTorch reference tensors can be written as fixtures the
C++ parity harness consumes. Writes C-order float32.
"""

import struct


def write_npy(path, shape, floats):
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }" % (
        ", ".join(str(int(d)) for d in shape) + ("," if len(shape) == 1 else "")
    )
    # Pad header so that 10 (preamble) + len(header) is a multiple of 64, ending '\n'.
    preamble = 10
    pad = (64 - (preamble + len(header) + 1) % 64) % 64
    header = header + " " * pad + "\n"

    with open(path, "wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))                 # version 1.0
        f.write(struct.pack("<H", len(header)))
        f.write(header.encode("ascii"))
        f.write(struct.pack("<%df" % len(floats), *floats))


if __name__ == "__main__":
    import sys
    # gen_sample_npy.py <path>: a known 2x3 array.
    write_npy(sys.argv[1], [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
