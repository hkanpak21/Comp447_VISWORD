# issue-01 — Legible cropping (native resolution + text-aware heuristics)

**Owner:** US · **Slice:** E1 · **Status:** needs-triage

## What to build
A cropping path that keeps text legible and each tile meaningful. Cut each 980×980 page
into tiles at **native resolution** (`crop_size == target_size`, no downsample — this is
the fix for the 2.19× shrink that made body text illegible). Add heuristics so each tile
is a clean unit of text: drop blank / very-low-text tiles, drop tiles that contain only
a fragment of a line, and snap tile boundaries to the whitespace gaps between text lines
(e.g. via a horizontal projection profile or light layout detection). Emit a sample of
crops to eyeball and a fresh evaluation slice disjoint from training pages.

## Acceptance criteria
- [ ] Returned tiles are at native resolution (no shrink): verified on synthetic pages.
- [ ] Blank/low-text tiles and fragment/half-line tiles are excluded.
- [ ] Tile boundaries fall in inter-line gaps (no glyph row is sliced).
- [ ] Sample crops are visibly legible (operator eyeball check) and a new disjoint eval slice is produced.
- [ ] Cropping unit tests pass (native-res, rejection, snapping, coverage).

## Blocked by
- issue-00 (backup before changing the data pipeline).

## Notes / scope (added during implementation — `TextAwareCropper`)
- The new `TextAwareCropper` lives alongside the untouched `NonOverlappingCropper` in
  [cropper.py](../../../src/visword/data/cropper.py) (additive). `target_size` defaults to
  `crop_size` → no downsample; the secondary high-res path sets them unequal.
- **Line-snapping is vertical (y-axis) only.** Inter-line gaps are found via a horizontal
  projection profile and tile y-boundaries snap into them, so no glyph *row* is sliced.
  Horizontal (x-axis) tiling still cuts a wide text column at native `crop_size` strides
  (words can be split left/right); this is in scope per D16 (native-224 grid for all encoders).
- **Fragment exclusion is best-effort.** Blank/low-text tiles are dropped (`min_text_ratio`),
  and gap-snapping prevents half-lines for normal body text (lines ≪ `crop_size`). A single
  line *taller* than `crop_size` has no gap to snap to and is hard-cut (test-pinned); this
  edge does not occur for Wikipedia body text but the AC wording "fragment excluded" is
  best-effort, not absolute.
