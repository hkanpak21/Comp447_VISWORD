# V1 — Issue index

Task tickets for [PRD1.md](../PRD1.md). See [../README.md](../README.md) for the plain
one-pager and [../RESULTS.md](../RESULTS.md) for the append-only results ledger.

**Owners:** `[US]` = hkanpak21's lane · `[BARIŞ]` = delegated to Barış (he reads the repo).
The two lanes are **non-overlapping**: Barış owns the **I-JEPA** Text-Target reader (he
has the trained checkpoint, which his account can't share); we own everything else,
including **our own reader on a pretrained MAE autoencoder** (`facebook/vit-mae-*`).

**Hard rules (from [/CLAUDE.md](../../../CLAUDE.md)):** additive only; never delete prior
work; deletions/git-rewrites need operator OK; accumulate numbers in RESULTS.md.

| # | Ticket | Owner | Blocked by |
|---|---|---|---|
| 00 | Reconcile repo + back up + prefetch models | US | — |
| 01 | Legible cropping (native res + text-aware) | US | 00 |
| 02 | Re-baseline comparison grid @ legible res (+ MAE) | US | 01 |
| 03 | Add document-pretrained family (Pix2Struct/Donut/Nougat/ColPali) | US | 02, 00 |
| 04 | Our reader: text-target on **MAE**, body target, page eval | US | 01, 00 |
| 05 | Perfect-text upper bound | US | 02 |
| 06 | Attention "where it reads" | US | 02 |
| 07 | Confound control (random title-masking) | US | 04 |
| 08 | Global-page token / two-stream (optional) | US | 02 |
| 09 | High-frequency autoencoder arm (optional) | US | 04 |
| 10 | Finish **I-JEPA** Text-Target reader heads (MLP/SALAD 30k) | BARIŞ | — |
| 11 | I-JEPA body-target variant (optional, parity w/ 04) | BARIŞ | — |

Dependency order for our lane: **00 → 01 → {02 → 03,05,06,08; 04 → 07,09}**.
Barış's lane (10, 11) runs in parallel; he already has the checkpoint.
