# boltz.cpp — component status & path to completion

Honest mapping of the Python inference pipeline to the C++/ggml port. "Validated"
means an automated C++ test passes (equivalence vs an independent reference, or
analytic values) on the ggml CPU backend. No trained weights are present in the
build environment, so nothing here is yet checked for *numeric parity with the
real model* — that requires the checkpoints (see "Blockers").

## Test suites (29 ctest cases, all passing)
Run: `cmake -S cpp -B cpp/build && cmake --build cpp/build && ctest --test-dir cpp/build`

## Component map

| Python component | C++ | Status |
|---|---|---|
| `parse.schema` residue constraints | `residue_constraints` | ✅ validated (ports both pytest files) |
| `inverse_fold.build_constraint_logit_mask` | `residue_constraints` | ✅ validated |
| `diffusion` schedule + Karras precond | `diffusion_schedule` | ✅ validated (analytic) |
| Linear / LayerNorm / GELU / SiLU / BatchNorm(eval) | `ggml_nn` | ✅ validated (ggml vs scalar) |
| `transition.Transition` (SwiGLU) | `ggml_nn::transition` | ✅ validated |
| `GaussianSmearing` | `featurizer` | ✅ validated |
| `scatter_utils` (sum/max/softmax) | `gnn` | ✅ validated |
| `init_knn_graph` (single batch) | `gnn::knn_graph` | ✅ validated |
| `MLPAttnGNN` encoder layer | `gnn_layer` | ✅ validated (ggml MLPs + scatter) |
| gguf weight load | `weight_store` | ✅ validated (round-trip) |
| `.ckpt → .gguf` converter | `tools/convert_ckpt_to_gguf.py` | ⚠️ format validated (cross-lang round-trip); conversion needs torch+ckpt |
| mmCIF `_atom_site` read | `cif` | ✅ validated (parses real 1brs.cif) |
| **Inverse-folding model end-to-end** | `ifold_model` + `boltzcpp_ifold` | ✅ **runs on real example** (`ifold_example_1brs`); simplified embedder + greedy head; synthetic weights |
| `triangular.TriangleMultiplication` (out/in) | `triangle_mult` | ✅ validated (wiring); needs golden-tensor parity check |
| `attention.AttentionPairBias` | `attention_pair_bias` | ✅ validated (wiring) |
| `outer_product_mean.OuterProductMean` | `outer_product_mean` | ✅ validated (wiring) |
| `triangular_attention` (start/end node) | `triangle_attention` | ✅ wiring-validated (AF2 bias broadcasting documented; needs golden-tensor parity) |
| Pairformer block | `pairformer` | ✅ wiring-validated (block vs op composition) |
| Full Pairformer trunk (stack of blocks) + distogram head | `trunk` | ✅ validated |
| MSA module | — | ❌ |
| Diffusion transformer (AdaLN, conditioned transition, fourier) | `diffusion_transformer`, `diffusion_blocks` | ✅ validated |
| Reverse-diffusion sampler loop (EDM/AF3) | `diffusion_sampler` | ✅ validated (oracle convergence) |
| Atom attention key-gather (single_to_keys) | `atom_windowing` | ✅ validated |
| Windowed atom attention (cross-attn + gather + mask) | `atom_transformer` | ✅ validated |
| Diffusion module end-to-end (score-net + sampler) | `test_diffusion_pipeline` | ✅ runs (linear atom embed/decode placeholders; synthetic weights) |
| Confidence heads (pae/pde/plddt/resolved + aggregation) | `confidence` | ✅ validated |
| Affinity head (cross-pair pool + ReLU MLPs) | `affinity` | ✅ validated |
| Featurizer: RelativePositionEncoder (rel pos features) | `rel_pos` | ✅ validated |
| Featurizer: token features (res-type one-hot, mol_type, masks) | `featurizer`/`const` | ✅ validated |
| Featurizer: atom chemistry (RDKit conformers, CCD, MSA) | — | ❌ external (RDKit + mols data) |
| mmCIF writer (output) | `mmcif_writer` | ✅ validated (write→parse round-trip) |
| Folding-style demo (trunk + diffusion + confidence) | `fold_demo` / `boltzcpp_fold` | ✅ runs on real 1brs.cif (synthetic weights; simplified atom featurization) |
| Design end-to-end + real-weight folding/affinity | — | ❌ (needs atom featurizer + weights) |

## What "runs" today
`boltzcpp_ifold <model.gguf> example/inverse_folding/1brs.cif A` parses the real
CIF, builds the KNN graph, runs the 6-layer ggml encoder and prints a sequence.
With a synthetic gguf the output is not meaningful (random weights); the pipeline
**executing** end to end is the demonstrated result.

## Blockers to "the entire pipeline, run all examples with parity"
1. **Trained weights (~6GB, 5 checkpoints).** Not in the build env; can't be
   downloaded here (no torch, no HF cache). Needed to: run the real converter,
   produce golden tensors, and validate numeric parity / get meaningful output.
2. **GPU + Vulkan SDK.** Dev/validation runs on ggml's CPU backend; the Vulkan
   backend can't be built or run here. Triangle ops are also the custom kernels
   that most need a real Vulkan implementation + a GPU to benchmark.

## Next steps (in order)
1. With weights available: run `convert_ckpt_to_gguf.py` on `boltzgen1_ifold.ckpt`
   and a golden-tensor dump; close numeric parity on the inverse-folder.
2. Port `triangular_attention` (validate against golden tensors), assemble the
   Pairformer block + MSA module → trunk.
3. Diffusion module network + sampler loop → folding end-to-end → design.
4. Full featurizer + mmCIF writer; confidence/affinity heads.
5. Move hot kernels (triangle mult/attn, pair-bias attn) to Vulkan; benchmark.
