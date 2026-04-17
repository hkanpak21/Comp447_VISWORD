# VisWord-SALAD — Acceptance Tests

Each section lists the tests the agent must implement and pass before
marking a phase complete. Tests are written with `pytest`. Markers:

- no marker → pure-Python, CPU, fast (<5 s each)
- `@pytest.mark.integration` → small-scale end-to-end, may need disk cache
- `@pytest.mark.gpu` → requires CUDA; skip if `torch.cuda.is_available()` is False
- `@pytest.mark.slurm` → requires SLURM, not run by `pytest` but listed here

---

## Phase A — Skeleton and data pipeline

### A1 `tests/test_salad_bridge.py`
- `test_bridge_imports_official_classes`: `OfficialSALAD` and
  `OfficialDINOv2` are importable from `visword.models.salad_bridge`
  and come from `third_party/salad/`, verified via `Path(module.__file__).resolve()`.
- `test_bridge_fails_cleanly_without_vendored_repo`: if
  `third_party/salad/` is missing, the import raises `RuntimeError` with
  a message containing `scripts/vendor_salad.sh`.
- `test_salad_forward_shape`: build `OfficialSALAD(num_channels=768,
  num_clusters=64, cluster_dim=128, token_dim=256)`, feed a synthetic
  `(patches=(B,768,16,16), cls=(B,768))` tuple, assert output shape is
  `(B, 64*128 + 256)` = `(B, 8448)` and is L2-normalised (norm ≈ 1).

### A2 `tests/test_manifest.py`
- `test_manifest_roundtrip`: write a manifest with 3 fake rows, read it
  back, assert identical.
- `test_fingerprint_detects_tampering`: write cache, record fingerprint,
  mutate one PNG, recompute fingerprint, assert mismatch.
- `test_manifest_partial_resume`: simulate a prefetch interrupted after
  row 5 out of 10; invoke the writer with `--resume`; final manifest
  has 10 rows, and rows 0-4 have their original `image_sha256`.

### A3 `tests/test_cropper.py`
- `test_non_overlapping_crops`: on a synthetic 980×980 image with
  `crop_size=490, overlap=0`, expect exactly 4 crops that each occupy a
  distinct 490×490 quadrant. Verify zero pixel overlap via bbox maths.
- `test_min_text_ratio_filter`: a fully-white image yields 1 crop (the
  fallback centre crop); an all-dark image yields all crops (all above
  threshold).
- `test_target_size_resize`: regardless of `crop_size`, every returned
  crop is `(target_size, target_size)`.

### A4 `tests/test_config.py`
- `test_config_merge`: loading `debug.yaml` on top of `default.yaml` with
  a `--set train.epochs=2` override produces the expected merged object.
- `test_config_hash_deterministic`: same config → same hash, different
  config → different hash, order of keys in YAML doesn't affect the hash.

### A5 `@pytest.mark.slurm` `tests/slurm_smoke/test_prefetch.md`
Not a pytest; a documented manual test. After submitting `prefetch.sbatch`
with `target-rows=50`:
- SLURM exit status 0.
- `$DATA_DIR/wiki_ss/manifest.json` exists and has 50 rows.
- `$DATA_DIR/wiki_ss/blobs/00/0000000.png` opens as a valid PNG.
- `.fingerprint` matches `sha256(sorted rows serialised)`.

---

## Phase B — Model wrappers

### B1 `tests/test_dinov2_salad.py` `@pytest.mark.gpu`
- `test_model_forward_shape`: `DINOv2SALAD(cfg)` on a `(2, 3, 224, 224)`
  CUDA tensor returns `(2, 8448)`, L2-normalised.
- `test_trainable_param_count`: matches the paper's "last 4 blocks
  trainable + aggregator" count (within 5 % tolerance to allow for minor
  impl differences).
- `test_model_state_dict_roundtrip`: save, load, forward — outputs
  exactly equal (`torch.equal`).

### B2 `tests/test_dinov2_cls.py` `@pytest.mark.gpu`
- `test_cls_baseline_forward_shape`: returns `(B, 256)`, L2-normalised.

### B3 `tests/test_losses.py`
- `test_infonce_multi_positive_is_reasonable`: random embeddings with
  known labels; loss is in a reasonable range (between 1.0 and 10.0 for
  B=32 fully random).
- `test_infonce_zero_when_perfect`: labels-clustered embeddings (same-
  class points identical, separated by far distance from other classes);
  loss ≈ 0.
- `test_multisim_runs`: just import and execute one forward + backward.

### B4 `tests/test_batch_stats.py`
- `test_batch_report_fields`: output JSON has all required keys;
  `positives_per_query_mean` equals `k_per_page - 1` for a synthetic
  dataset with distinct pages.

---

## Phase C — End-to-end training

### C1 `tests/test_train_debug.py` `@pytest.mark.integration @pytest.mark.gpu`
- Create a tiny cache of 50 pages offline (fixture).
- Run `python -m visword.train --config configs/debug.yaml --set
  data.num_train_samples=40 data.num_eval_samples=10 train.epochs=1`.
- Assert:
  - A run directory was created under `runs/`.
  - `config.resolved.yaml`, `provenance.json`, `metrics.jsonl`,
    `checkpoints/last.pt` all exist.
  - `metrics.jsonl` has at least one line per step; schema matches §6.
  - Loss at step 0 is finite and positive.
  - No file was written outside `runs/`.

### C2 `tests/test_offline_mode.py` `@pytest.mark.integration`
- Set `HF_HUB_OFFLINE=1`, run training with an *empty* cache → training
  must abort with a clear message pointing to the prefetch script. It
  must not silently try to download.

### C3 `tests/test_run_dir_contract.py`
- After a successful debug run, every file listed in `PROJECT_SPEC.md §11`
  is present.

---

## Phase D — Evaluation

### D1 `tests/test_eval_phase1.py` `@pytest.mark.integration @pytest.mark.gpu`
- With a debug-trained checkpoint, run `python -m visword.eval_phase1
  --run-dir <path>`.
- Assert `phase1_recall.json` is written with the §7 schema.
- `recall@20 >= recall@10 >= recall@5 >= recall@1` (monotonicity).
- `sanity.gap > 0` for a trained model.

### D2 `tests/test_eval_phase2.py` `@pytest.mark.integration @pytest.mark.gpu`
- Analogous to D1 but for Phase 2. Requires anchors cache populated.

### D3 `tests/test_eval_runs_without_training_code_loaded.py`
- Verify `eval_phase1.py` doesn't import `train.py` (keeps surfaces
  decoupled). Use `sys.modules` introspection after import.

---

## Phase E — Interpretability

### E1 `tests/test_interpret_attention.py` `@pytest.mark.integration @pytest.mark.gpu`
- Produces a non-empty `attention_sample0.png` file and a JSON sidecar
  with mean/max attention stats.

### E2 `tests/test_interpret_salad_internals.py` `@pytest.mark.integration @pytest.mark.gpu`
- `interpret/salad_hooks.json` identifies a submodule for each of
  `score`, `cluster_features`, `token_features`.
- Sinkhorn post-processing produces a doubly-stochastic matrix (row
  sums, column sums each within `1e-3` of uniform).
- Dustbin mass total ∈ `[0, 1]`.

### E3 `tests/test_interpret_cls_vs_vlad.py` `@pytest.mark.integration @pytest.mark.gpu`
- For a trained checkpoint, the sum of same-page VLAD-half cosine +
  same-page CLS-half cosine ≈ (within `1e-4` of) the same-page full
  cosine. (Validates our disjoint-dimension slicing is correct.)

### E4 `tests/test_interpret_patch_neighbours.py` `@pytest.mark.integration @pytest.mark.gpu`
- Produces `k_examples` PNGs. Each file is readable as an image and has
  non-trivial pixel variance (not all one colour).

---

## Cross-cutting

### X1 `tests/test_no_internet_at_runtime.py`
- Monkeypatch `socket.socket` to raise; run `python -m visword.train
  --config configs/debug.yaml` (with cache populated). Must succeed.

### X2 `tests/test_determinism.py` `@pytest.mark.integration @pytest.mark.gpu`
- Two debug runs with same config + seed produce identical `metrics.jsonl`
  for at least the first 10 steps (within `1e-5` tolerance on loss).
  If not: investigate before proceeding (non-determinism is often a
  CUDA ops flag, not a real bug).

### X3 `tests/test_summarise_run.py`
- `scripts/summarise_run.py <run-dir>` on a completed debug run
  produces output containing: `experiment_name`, final loss, final
  recall@10, SALAD commit SHA, git SHA, wall time.

---

## Test invocation examples

```bash
# CPU-only, fast — can run on laptop
pytest -x -q

# With GPU, excluding slow integration
pytest -x -q -m "gpu and not integration"

# Full integration suite (needs cache and GPU)
DATA_DIR=/tmp/visword-test-cache pytest -x -q -m "integration"
```
