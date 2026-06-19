"""Minimal, dependency-free GGUF writer (Python stdlib only).

Implements just enough of the GGUF v3 container to serialize fp32 tensors plus
string/uint32 metadata, matching what the C++ `boltz::WeightStore` (ggml's gguf
loader) reads back. No numpy, no `gguf` package — consistent with the project's
"hand-roll everything except ggml" constraint.

GGUF layout (little-endian):
  magic "GGUF" | version:u32 | tensor_count:u64 | kv_count:u64
  kv*:    key(str) | type:u32 | value
  info*:  name(str) | n_dims:u32 | dims[u64..] | ggml_type:u32 | offset:u64
  <pad to alignment> | tensor data (each padded to alignment)
where a gguf string is u64 length + utf-8 bytes, and tensor `offset` is relative
to the start of the (aligned) data section.
"""

import struct

_MAGIC = b"GGUF"
_VERSION = 3
_ALIGN = 32

# GGUF metadata value types.
_T_UINT32 = 4
_T_FLOAT32 = 6
_T_STRING = 8

_GGML_TYPE_F32 = 0


def _align(n, a=_ALIGN):
    return (n + a - 1) // a * a


def _pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


class GGUFWriter:
    def __init__(self):
        self._kv = []       # (key, type, raw_value_bytes)
        self._tensors = []  # (name, ne_dims, data_bytes)

    def add_str(self, key, value):
        self._kv.append((key, _T_STRING, _pack_str(value)))

    def add_u32(self, key, value):
        self._kv.append((key, _T_UINT32, struct.pack("<I", value)))

    def add_tensor(self, name, ne_dims, floats):
        """ne_dims in ggml order (ne0 fastest). `floats` is a flat row-major list
        whose layout already matches ne_dims (ne0 contiguous)."""
        data = struct.pack("<%df" % len(floats), *floats)
        self._tensors.append((name, list(ne_dims), data))

    def write(self, path):
        # Tensor offsets depend only on data sizes + alignment.
        offsets = []
        off = 0
        for _, _, data in self._tensors:
            offsets.append(off)
            off = _align(off + len(data))

        out = bytearray()
        out += _MAGIC
        out += struct.pack("<I", _VERSION)
        out += struct.pack("<Q", len(self._tensors))
        out += struct.pack("<Q", len(self._kv))

        for key, vtype, vbytes in self._kv:
            out += _pack_str(key)
            out += struct.pack("<I", vtype)
            out += vbytes

        for (name, ne_dims, data), offset in zip(self._tensors, offsets):
            out += _pack_str(name)
            out += struct.pack("<I", len(ne_dims))
            for d in ne_dims:
                out += struct.pack("<Q", d)
            out += struct.pack("<I", _GGML_TYPE_F32)
            out += struct.pack("<Q", offset)

        # Pad to the data section, then write each tensor padded to alignment.
        out += b"\x00" * (_align(len(out)) - len(out))
        for (_, _, data), offset in zip(self._tensors, offsets):
            out += data
            out += b"\x00" * (_align(len(data)) - len(data))

        with open(path, "wb") as f:
            f.write(out)
