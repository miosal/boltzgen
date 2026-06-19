# boltz.cpp — component status & path to completion

Honest mapping of the Python inference pipeline to the C++/ggml port. "Validated"
means an automated C++ test passes (equivalence vs an independent reference, or
analytic values) on the ggml CPU backend.

**Trained-weight numeric parity (inverse-fold GNN core): DONE.** The
inverse-folding encoder (`inverse_folding_encoder`: linear_token_to_node,
linear_token_to_pair, 6× MLPAttnGNN) and decoder (`structure_module`: seq_to_s,
3× MLPAttnGNNDecoder, predictor) — 314 of the checkpoint's 357 tensors — are now
checked for *numeric parity against the PyTorch reference on the real
`boltzgen1_ifold.ckpt` weights* (`test_parity`). Measured on a fixed-seed
scenario (N=24, E=288): encoder s max_abs **4.65e-4** (ref std ~4.0), encoder z
max_abs **2.24e-4** (ref std ~5.1), decoder logits max_abs **3.36e-4** on values
up to 291 (rel **1.2e-6**), and **0/24** per-position argmax mismatches (the
greedy designed token matches the reference exactly). The ~1e-4 gap is fp32
matmul reassociation between ggml and PyTorch across 9 GNN layers. Inputs
(`s_inputs`, edge features, the post-RNG sequence-visibility one-hot) are dumped
from the reference (`tools/dump_golden.py`); see "Remaining gaps" for the parts
NOT yet covered (the upstream InputEmbedder, the hand-rolled featurizer wiring to
real data, and the stochastic autoregressive sampler).

## Test suites (37 ctest cases, all passing; 5 example pipelines run)
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
| `.ckpt → .gguf` converter | `tools/convert_ckpt_to_gguf.py` | ✅ converts the real `boltzgen1_ifold.ckpt` (loads w/o omegaconf via a permissive unpickler; shortens names < ggml's 64-char limit; drops unused input_embedder) |
| **Inverse-fold encoder+decoder, REAL trained weights** | `ifold_real` + `test_parity` | ✅ **numeric parity vs PyTorch** on `boltzgen1_ifold.ckpt` (see header for measured diffs; argmax exact) |
| mmCIF `_atom_site` read | `cif` | ✅ validated (parses real 1brs.cif) |
| Inverse-folding model end-to-end (demo) | `ifold_model` + `boltzcpp_ifold` | ✅ runs on real example (`ifold_example_1brs`); **simplified** embedder + greedy head on synthetic weights (the real-weight path is `ifold_real`, above) |
| `triangular.TriangleMultiplication` (out/in) | `triangle_mult` | ✅ validated (wiring); needs golden-tensor parity check |
| `attention.AttentionPairBias` | `attention_pair_bias` | ✅ validated (wiring) |
| `outer_product_mean.OuterProductMean` | `outer_product_mean` | ✅ validated (wiring) |
| `triangular_attention` (start/end node) | `triangle_attention` | ✅ wiring-validated (AF2 bias broadcasting documented; needs golden-tensor parity) |
| Pairformer block | `pairformer` | ✅ wiring-validated (block vs op composition) |
| Full Pairformer trunk (stack of blocks) + distogram head | `trunk` | ✅ validated |
| MSA featurization: single-sequence + multi-sequence-from-alignment (profile/deletions) | `featurizer` | ✅ validated (logic; real MSA *database files* are data-gated) |
| `InputEmbedder` token-level (res-type + profile + conditioning) | `input_embedder` | ✅ validated (vs scalar ref) |
| Diffusion transformer (AdaLN, conditioned transition, fourier) | `diffusion_transformer`, `diffusion_blocks` | ✅ validated |
| Reverse-diffusion sampler loop (EDM/AF3) | `diffusion_sampler` | ✅ validated (oracle convergence) |
| Atom attention key-gather (single_to_keys) | `atom_windowing` | ✅ validated |
| Windowed atom attention (cross-attn + gather + mask) | `atom_transformer` | ✅ validated |
| Diffusion module end-to-end (score-net + sampler) | `test_diffusion_pipeline` | ✅ runs (linear atom embed/decode placeholders; synthetic weights) |
| Confidence heads (pae/pde/plddt/resolved + aggregation) | `confidence` | ✅ validated |
| Affinity head (cross-pair pool + ReLU MLPs) | `affinity` | ✅ validated |
| Featurizer: RelativePositionEncoder (rel pos features) | `rel_pos` | ✅ validated |
| Featurizer: token features (res-type one-hot, mol_type, masks) | `featurizer`/`const` | ✅ validated |
| Featurizer: standard-residue atom topology (names/elements/atom-to-token/backbone) | `atom_featurizer` | ✅ validated (baked, no RDKit) |
| Featurizer: SMILES → ligand atom/bond graph | `smiles` | ✅ validated (8 cases) |
| Featurizer: 3-D conformer generation (distance-geometry/PBD) | `conformer` | ✅ validated (geometry: bond lengths, no clashes, deterministic) |
| mmCIF writer (output) | `mmcif_writer` | ✅ validated (write→parse round-trip) |
| Inverse-folding pipeline | `boltzcpp_ifold` | ✅ runs on real 1brs.cif |
| Design pipeline (real featurizers → embed → trunk → diffusion) | `boltzcpp_design` | ✅ runs on real 1brs.cif |
| Folding-style pipeline (trunk + distogram + confidence + affinity + diffusion) | `boltzcpp_fold` | ✅ runs on real 1brs.cif |
| Ligand pipeline (SMILES → conformer → mmCIF) | `boltzcpp_ligand` | ✅ runs on aspirin SMILES (arbitrary chemistry) |
| Diffusion module pipeline | `test_diffusion_pipeline` | ✅ runs |
| Arbitrary-ligand conformer *matching RDKit ETKDGv3+UFF exactly* | — | ◐ approximate embedding written; exact RDKit parity needs CCD/torsion data (HF) |
| Real MSA database + taxonomy pairing | — | ❌ data-gated (MSA files on HuggingFace) |
| Trained-weight numeric parity (inverse-fold GNN core) | `ifold_real` + `test_parity` | ✅ done — encoder+decoder match PyTorch on real weights |
| Trained-weight parity (InputEmbedder, full featurizer, AR sampler, other 4 models) | — | ◐ not yet — see "Remaining gaps" |

## What "runs" today (5 pipelines, on real inputs)
- `boltzcpp_ifold model.gguf example/inverse_folding/1brs.cif A` — inverse folding
- `boltzcpp_design example/inverse_folding/1brs.cif A` — featurize → embed → trunk → diffusion
- `boltzcpp_fold example/inverse_folding/1brs.cif A` — trunk + distogram + confidence + affinity + diffusion
- `boltzcpp_ligand "CC(=O)Oc1ccccc1C(=O)O"` — SMILES → 3-D conformer → mmCIF (arbitrary chemistry)
- `test_diffusion_pipeline` — diffusion schedule + score-net + sampler
- `test_parity` — **real** `boltzgen1_ifold.ckpt`: C++ inverse-fold encoder+decoder
  vs PyTorch reference (numeric parity; see header for measured diffs)

The 5 example *pipelines* still run on synthetic weights (so their output is not
numerically meaningful) — the pipelines **executing** end to end on real inputs
is the demonstrated result there. Numeric parity on *trained* weights is
demonstrated by `test_parity` for the inverse-fold GNN core.

## Reproducing the inverse-fold parity (HuggingFace egress now enabled)
```
pip install torch huggingface_hub                       # CPU torch is enough
python3 - <<'PY'                                         # fetch the checkpoint
from huggingface_hub import hf_hub_download
print(hf_hub_download("boltzgen/boltzgen-1","boltzgen1_ifold.ckpt",repo_type="model"))
PY
python3 cpp/tools/convert_ckpt_to_gguf.py <ckpt> cpp/build/boltzgen1_ifold.gguf --arch boltzgen-ifold
python3 cpp/tools/dump_golden.py        <ckpt> cpp/tests/fixtures/ifold   # fixtures committed
cmake -S cpp -B cpp/build && cmake --build cpp/build -j && ctest --test-dir cpp/build -R test_parity
```
The golden `.npy` fixtures are committed; the 12 MB gguf (derived from the gated
checkpoint) is not — `test_parity` skips when it is absent, runs and asserts
parity when present.

## Remaining gaps
The inverse-fold **GNN core** now has measured trained-weight parity (above).
What is still NOT covered for a *full end-to-end inverse-fold sequence on real
weights*, and for the other models:
1. **InputEmbedder + featurizer wiring + autoregressive sampler.** The encoder's
   node features `s_inputs` come from the upstream `InputEmbedder` (a 43-tensor
   atom-attention encoder) run over featurizer output; the decoder's published
   inference is the stochastic `sample()` (randperm + temperature multinomial),
   not the deterministic `forward()` validated here. Porting the InputEmbedder to
   real weights + wiring the hand-rolled featurizer to real mol data + an
   inject-dumped-RNG sampler would make `boltzcpp_ifold` reproduce the Python
   designed sequence end to end. (The featurizer/atom-embedder algorithms are
   already written and unit-tested on synthetic weights; this is integration +
   the larger InputEmbedder weight port, not new algorithms.)
2. **Other 4 checkpoints (design/fold/affinity).** `convert_ckpt_to_gguf.py` now
   loads real Lightning checkpoints; extend the `ifold_real`-style real-weight
   wiring + golden dumps to the Pairformer trunk / diffusion / confidence /
   affinity to get parity there too (larger; the long-name shortening rule in the
   converter will need per-arch entries).
2. **Real MSA database + CCD/mols data.** Also HF-only. The per-alignment MSA
   featurization and ligand topology/conformer logic are written; pulling the
   reference *databases* is data-gated. The conformer generator is an
   approximate distance-geometry embedding, not bit-identical to RDKit ETKDGv3.
3. **GPU + Vulkan SDK.** Dev/validation runs on ggml's CPU backend; building/
   benchmarking the Vulkan backend needs a GPU (none here).
