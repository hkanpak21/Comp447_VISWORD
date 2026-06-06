# issue-10 — Finish the I-JEPA Text-Target reader heads (MLP/SALAD 30k)

**Owner:** BARIŞ · **Slice:** E4 (Barış's lane) · **Status:** needs-triage

## What to build
Complete the retrieval-head fine-tunes on top of the I-JEPA Text-Target checkpoint that
already exists in Barış's scratch (this is Barış's lane because the checkpoint is in his
account and can't be shared due to Valar permissions). Run the two configs already in the
repo — `configs/ijepa_text_target_mlp_30k.yaml` and `configs/ijepa_text_target_salad_30k.yaml`
(using `models/ijepa_salad.py`) — and fill in the results that are currently marked **TBD**
in the paper's tables.

## Acceptance criteria
- [ ] Page/Protocol-A recall@k for I-JEPA-Text-MLP-30k and I-JEPA-Text-SALAD-30k.
- [ ] The TBD rows in the paper tables are filled.
- [ ] Numbers appended to RESULTS.md (the rows currently marked "TBD — Barış running").

## Blocked by
- None — Barış already has the checkpoint and configs.
