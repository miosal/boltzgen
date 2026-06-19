# BoltzGen → C++ / Vulkan Inference Port — Plan

Status: design proposal. Goal: a self-contained, cross-platform (Vulkan) C++
inference engine for BoltzGen, "llama.cpp for Boltz". This document scopes the
work, calls out the real blockers, and proposes a staged plan with validation
gates.

---

## 0. Reality check — how this differs from llama.cpp

llama.cpp is tractable because an LLM is *one* model with a *tiny* op set
(GEMM, RMSNorm, RoPE, SwiGLU, softmax-attention), a *trivial* input pipeline
(BPE tokenizer over text), and a *cheap* per-step loop (one token, with a
KV-cache). BoltzGen breaks every one of those assumptions:

| Axis | llama.cpp (LLM) | BoltzGen |
| --- | --- | --- |
| Models | 1 transformer | 5: design-diffusion, inverse-fold GNN, folding (Boltz-2), confidence, affinity |
| Core blocks | decoder attention + MLP | Pairformer (triangle mult + triangle attention + pair-bias attn), MSA module, atom-level windowed attention, diffusion module, GNN |
| Compute shape | O(N²) attn, O(N) decode steps | **O(N³)** triangle ops × 64 blocks × 3 recycles; **500** diffusion steps each a full network eval |
| Memory trick | KV-cache | none — full O(N²) pair tensor (token_z=128) every block |
| Input pipeline | BPE tokenizer | CIF/PDB parse (gemmi), ligand/CCD + 3-D conformers (RDKit ETKDGv3+UFF), MSA pairing (numba), DSSP, featurization (~6k LOC) |
| "6GB model" | one file | **5 separate checkpoints**, loaded per-step |

Two consequences:

1. **The preprocessing is a co-equal boss fight, not a footnote.** ~6000 lines
   of non-NN Python depend on `gemmi`, `rdkit`, `numba`, `pydssp`,
   `pdbeccdutils`. **Project constraint: we hand-roll all of this in C++ — the
   only third-party dependency is `ggml` (git submodule, plain CMake
   `add_subdirectory`, no package manager).** That means hand-written CIF/PDB
   parsing, tokenization, featurization, MSA pairing, DSSP, and mmCIF writing.
   The one genuinely hard piece to hand-roll is RDKit's ETKDGv3 + UFF conformer
   generation for arbitrary ligands. We sidestep it: **bake static reference
   conformers for the standard residues** (20 AAs + nucleotides, fixed CCD
   ideal coordinates) into a data table generated offline, and **defer
   arbitrary-SMILES conformer generation** past the protein-only milestones. The
   first milestone (protein inverse folding) needs no conformer generation at
   all.

2. **This is ~6 genuinely custom GPU kernels on top of a standard set**, not a
   from-scratch backend. We should stand on an existing Vulkan tensor runtime.

So the plan is: **build on `ggml`'s Vulkan backend** (gives us GEMM, softmax,
norm, elementwise, gguf weight format, quantization, and cross-platform Vulkan
dispatch for free), add the Boltz-specific kernels, and **hand-roll the entire
preprocessing/I-O front-end in C++** (no gemmi/rdkit/yaml libs).

### Build/dependency constraints (hard requirements)
- **Only dependency: `ggml`**, vendored as a **git submodule**.
- **No package manager** (no vcpkg/conan/apt-for-deps). Plain CMake primitives
  only: `add_subdirectory(third_party/ggml)`, `target_link_libraries(... ggml)`.
- **Hand-roll everything else**: YAML-subset reader, CIF/PDB parser, tokenizer,
  featurizer, MSA reader, DSSP, mmCIF/PDB writer, gguf reader. C++17, STL only.
- Develop and validate first on **ggml's CPU backend** (no GPU needed); Vulkan
  is a backend toggle once kernels are validated.

---

## 1. Architecture facts (the port target)

Hyperparameters (large model, `config/train/boltzgen.yaml`):
- token_s=384, token_z=128, atom_s=128, atom_z=16
- Pairformer: 64 blocks, 16 heads; triangle attn: 4 heads × width 32
- Diffusion transformer: 24 layers, 16 heads; atom enc/dec depth 3
- Atom windowed attention: 32 queries / 128 keys per window
- Diffusion: sigma_min 4e-4, sigma_max 160, sigma_data 16, rho 7; 64 distogram bins
- Small model exists (12 pairformer blocks, depth 8) — **use it as the first
  laptop target**.

Inference settings (per-step configs):
- design: 500 sampling steps, 3 recycles, bf16-mixed
- folding: 200 steps, 5 samples, bf16-mixed
- inverse fold: fp32, autoregressive, topk=30 KNN, temp 0.1
- affinity: bf16-mixed, ensemble of 2

### Op inventory (what needs a Vulkan kernel)

Standard (ggml already has, or trivial):
- Linear / GEMM, LayerNorm (eps 1e-5), residual add, SiLU/GELU/sigmoid gates,
  softmax, one-hot, broadcasting, Fourier/Gaussian feature embeddings.

**Custom kernels to write (the real work):**
1. **Triangle multiplication** outgoing `bikd,bjkd->bijd` / incoming
   `bkid,bkjd->bijd`, with input LayerNorm, projection, sigmoid gate, output
   norm+gate. Run in **fp32** (reference disables autocast here).
   `model/layers/triangular.py`.
2. **Triangle attention** starting/ending node — attention along rows/cols of
   the pair tensor with a pair-derived bias. `model/layers/triangular_attention/`.
3. **Attention with pair bias** (`scaled_dot_product_attention` + additive bias
   from z, + AdaLN conditioning). `model/layers/attention.py`.
4. **Outer product mean** `bsic,bsjd->bijcd` + masked mean. MSA→pair.
   `model/modules/outer_product_mean.py`.
5. **Atom windowed/local attention** (queries-per-window 32, keys 128) for the
   diffusion atom encoder/decoder. `model/modules/encoders.py`, `diffusion.py`.
6. **Scatter/gather + KNN** for the inverse-folding GNN: `scatter_softmax`,
   `scatter_sum`, k-NN graph build. `model/modules/inverse_fold.py`.

`cuequivariance` kernels are **optional** — the pure-`torch.einsum` fallback is
the reference semantics and is what we port. We do not need cuEquivariance.

Diffusion sampler math (Karras preconditioning, step/noise schedules, centering,
random rigid augmentation, weighted rigid align) is mostly scalar/elementwise
and small — keep on **CPU**; only the network forward is GPU work.

---

## 2. Engine architecture

```
+------------------------------------------------------------+
|  Front-end (hand-rolled C++, STL only)                     |
|   YAML spec  → parse → tokenize → featurize → features      |
|   hand-written: yaml-subset, CIF/PDB parse, featurizer,     |
|   MSA reader, DSSP, mmCIF writer  (NO gemmi/rdkit)          |
+------------------------------+-----------------------------+
                               | in-memory feature tensors
                               v
+------------------------------------------------------------+
|  boltz.cpp engine (C++)                                     |
|   weights: gguf  (converted from .ckpt)                     |
|   graph builder per model (design / ifold / fold / conf /  |
|     affinity) + sampler loops on CPU                        |
+------------------------------+-----------------------------+
                               | ops
                               v
+------------------------------------------------------------+
|  Backend = ggml Vulkan  +  ~6 custom Boltz kernels (GLSL/   |
|   SPIR-V): triangle mult, triangle attn, pair-bias attn,    |
|   outer-product-mean, atom-window attn, scatter/knn         |
+------------------------------------------------------------+
                               | writes coords + scores
                               v
   Output: mmCIF via libgemmi (back-end), metrics CSV
```

Why `ggml`: Vulkan dispatch, memory pool, gguf, fp16/bf16/quantized GEMM, many
elementwise ops, and proven cross-platform builds — already done. We add a
custom-op extension for the 6 kernels above.

---

## 3. Methodology: golden-tensor oracle (non-negotiable)

These projects live or die on layer-by-layer numerical validation. Before any
Vulkan code:

1. Build a PyTorch dump harness: fix seed + a small fixed input, run reference
   inference, and serialize **every block-boundary tensor** (after input
   embedder, after each pairformer block, after MSA, after each diffusion step,
   GNN layers, heads) to `.npz`.
2. Every C++ kernel/graph stage is checked against its golden tensor with a
   tolerance budget (tighter fp32 for trunk/triangle; looser for bf16/fp16
   paths). A stage isn't "done" until it passes.
3. Diffusion/inverse-fold are stochastic (`torch.randn/randperm/multinomial`).
   Bit-exact cross-framework repro is infeasible; validate the **deterministic
   trunk exactly**, and the stochastic parts **statistically** (distribution of
   outputs, and exact match when fed identical injected noise from a dumped RNG
   stream).

---

## 4. Staged plan with gates

### Phase 0 — Scaffolding + oracle  (validate the toolchain on something trivial)
- Vendor/build `ggml` Vulkan backend; CMake skeleton; CI build for Linux + one
  laptop target.
- Weight converter `.ckpt → .gguf`: handle EMA prefix and the documented key
  renames (e.g. `token_transformer_layers.0.` → `token_transformer.`), dtype.
- PyTorch golden-tensor dump harness.
- **Gate:** reproduce the input embedder (or one pairformer block) in C++/Vulkan
  matching PyTorch within tolerance. Proves weight load + dispatch + numerics.

### Phase 1 — Inverse folding model end-to-end  (smallest, highest ROI first)
Rationale: fp32, no triangle attention, no diffusion. 6 encoder + 3 decoder GNN
layers + autoregressive sampling. Smallest surface that exercises the full
toolchain on a *useful* output.
- Kernels: gaussian smearing, KNN graph, scatter_softmax/sum, MLPAttnGNN,
  autoregressive multinomial sampling with aa-constraint masks + homomer tying.
- Input features supplied from Python front-end.
- **Gate:** given a fixed backbone, C++ recovers the same sequence logits as
  PyTorch (greedy/argmax exact; sampled distribution matches).

### Phase 2 — Pairformer trunk  (the custom-kernel core)
- Implement + validate kernels 1–4 (triangle mult, triangle attn, pair-bias
  attn, outer-product-mean) individually against golden tensors.
- Assemble MSA module + 64-block Pairformer + recycling + distogram head.
- Start with the **small** model (12 blocks) for fast iteration.
- **Gate:** full trunk distogram + pair/single outputs match reference.

### Phase 3 — Diffusion + folding end-to-end
- Kernel 5 (atom windowed attention); diffusion conditioning; 24-layer diffusion
  transformer; Karras sampler loop (CPU control, GPU forward); weighted rigid
  align; confidence heads.
- **Gate:** Boltz-2 folding of a sequence reproduces reference coords
  (statistically, with exact-noise check) + pLDDT/pAE within tolerance.

### Phase 4 — Design diffusion + affinity
- Design-specific conditioning (binding types, structure groups, secondary
  structure, contact conditioning), 500-step sampler, affinity head/ensemble.
- **Gate:** end-to-end `design → inverse_fold → fold` produces valid designs;
  spot-check metrics against the Python pipeline on a small example.

### Phase 5 — Ligand/small-molecule support + packaging
- Extend the hand-rolled featurizer to ligands: parse CCD components, use baked
  static conformers where available. **Arbitrary-SMILES conformer generation
  (ETKDGv3 + UFF) is the one deferred hard problem** — hand-rolling it is a
  sub-project; until then, support ligands only via supplied coordinates or
  precomputed conformer tables.
- Quantized (Q8/Q4) linear weights for footprint; single self-contained binary.

Note: because preprocessing is hand-rolled C++ from Phase 1 onward (not a Python
front-end), there is no separate "remove Python" step — Python is only ever used
for the offline build tools (ckpt→gguf converter, golden-tensor dumper, static
conformer table generator).

---

## 5. Cross-cutting risks & calls

- **Precision on Vulkan.** bf16 storage is patchy across consumer GPUs; fp16
  (`VK_KHR_16bit_storage`/`shaderFloat16`) is broader but numerically different.
  Plan: **fp16 storage + fp32 accumulate** for GEMM; **keep triangle ops and the
  diffusion sampler in fp32** (reference already forces fp32 there). Validate
  per-layer; do not assume bf16-mixed transfers cleanly.
- **Performance expectations.** Triangle attention is O(N³) × 64 blocks × 3
  recycles, and design runs 500 full diffusion evals. A laptop iGPU will be
  *minutes to tens of minutes per design*, not the A100 seconds in the README.
  Set this expectation up front. Quantization helps memory but the triangle ops
  (not plain GEMMs) dominate compute, so the llama.cpp-style quant speedup is
  smaller here. Crop/length limits (~256–512 tokens) are the practical envelope.
- **Memory.** Only the *active* step's checkpoint needs to be resident (~100MB
  small / ~400MB large in fp16) plus O(N²) pair activations (~tens of MB at 512
  tokens). The "6GB" is the sum of 5 models on disk, not peak RAM. Laptop-OK.
- **Determinism.** Document that exact cross-framework repro is out of scope;
  deliver validated-correct + statistically-faithful sampling, with an
  inject-dumped-noise mode for exact comparison during testing.
- **Scope discipline.** Don't port all 5 models at once. The phase order front-
  loads toolchain validation (Phase 1) and the hardest shared kernels (Phase 2)
  so later phases are mostly assembly.

---

## 6. Recommended first milestone

Phases 0 + 1 (scaffolding, weight converter, golden-tensor oracle, and the
inverse-folding model end-to-end). It validates the entire toolchain —
gguf weight loading, Vulkan dispatch, custom kernels, sampling — on the smallest
model, and produces a genuinely useful standalone tool (inverse folding a fixed
backbone) before we touch triangle attention or diffusion.
