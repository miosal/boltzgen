# boltz.cpp

A self-contained C++ / Vulkan inference engine for BoltzGen ("llama.cpp for
Boltz"). Design and rationale: [`../CPP_VULKAN_PORT_PLAN.md`](../CPP_VULKAN_PORT_PLAN.md).

## Constraints
- **Only dependency: `ggml`** (git submodule at `third_party/ggml`).
- **No package manager.** Plain CMake primitives only.
- Everything else — YAML-subset reader, CIF/PDB parser, tokenizer, featurizer,
  MSA reader, DSSP, mmCIF writer, gguf reader — is **hand-rolled C++17 / STL**.
- Develop/validate on the **ggml CPU backend** first; Vulkan is a build toggle.

## Build
```bash
git submodule update --init --recursive cpp/third_party/ggml
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
./cpp/build/boltzcpp_smoke      # prints "SMOKE OK" if ggml is wired correctly
```
Vulkan backend (requires a Vulkan SDK; off by default):
```bash
cmake -S cpp -B cpp/build -DBOLTZCPP_VULKAN=ON
```

## Status
- [x] Scaffold: ggml submodule, plain CMake, CPU smoke test (`src/smoke.cpp`).
- [ ] everything below.

## Roadmap (see plan doc for full detail)

### Phase 0 — toolchain + golden-tensor oracle
- [ ] `tools/convert_ckpt_to_gguf.py` — `.ckpt` state_dict → `.gguf`. Handle the
      EMA prefix and the documented key renames
      (`token_transformer_layers.0.` → `token_transformer.`).
- [ ] `tools/dump_golden.py` — fixed seed + fixed input, serialize every
      block-boundary tensor from the PyTorch reference to `.npz` for per-layer
      validation.
- [ ] `src/gguf_reader` — hand-rolled loader (or via ggml's gguf API).
- [ ] Validation harness: load a reference tensor, run the matching C++ stage,
      assert within tolerance.

### Phase 1 — inverse folding model (first milestone)
Smallest model: fp32, no triangle attention, no diffusion. Protein backbone →
sequence. Needs **no** ligand conformer generation.
- [ ] Front-end (hand-rolled): CIF/PDB parse → tokenize → featurize the
      protein inverse-folding inputs (backbone coords, masks, design mask,
      aa-constraint mask, KNN inputs). Static reference conformers for the 20
      standard residues baked from CCD offline.
- [ ] Kernels/graph: gaussian smearing, KNN graph build, `scatter_softmax` /
      `scatter_sum`, the MLPAttnGNN encoder (6 layers) + decoder (3 layers).
- [ ] Autoregressive sampling: random order, temperature (0.1) multinomial,
      aa-constraint masking, homomer tying.
- [ ] **Gate:** C++ reproduces PyTorch sequence logits (argmax exact; sampled
      distribution matches) on a fixed backbone.

### Later phases
Phase 2 Pairformer trunk (triangle mult/attn, outer-product-mean, pair-bias
attn) · Phase 3 diffusion + Boltz-2 folding · Phase 4 design diffusion +
affinity · Phase 5 ligand support + quantization/packaging.

## Layout
```
cpp/
  CMakeLists.txt        # ggml submodule + boltzcpp + smoke exe
  include/              # hand-rolled engine headers (added per phase)
  src/                  # engine + entry points
    smoke.cpp           # CPU-backend smoke test
  tools/                # offline Python build tools (converter, dumper)
  third_party/ggml/     # submodule — the only dependency
```
