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
