# VISWORD — Next Steps (master branch working doc)

> **Status: GRILLING CLOSED (Q1–Q16 resolved).** Next: PRD (`/to-prd`) → issues
> (`/to-issues`) → execution (P0 → E1). This doc stays the canonical alignment
> record; the PRD/issues operationalize it.
>
> Tags used below: **[CONFIRMED]** you verified · **[ASSUMED]** my read of the
> meeting notes, unverified · **[OPEN]** needs your answer · **[DECIDED]** locked
> after grilling.
>
> Last updated: end of grilling (Q1–Q16 resolved; grounded in Valar code/data/infra).

---

## ⚠ Preservation & safety — HARD RULES (read first, apply always)

1. **Never delete or overwrite Barış's work.** This includes the paper
   (`paper/visword_report.tex` + figures), his I-JEPA code, configs, SLURM scripts,
   and any `runs/` results on Valar. The paper stays exactly as-is unless explicitly
   asked to edit it.
2. **Any deletion or destructive change to prior work requires explicit confirmation
   from the operator first** — never delete, force-overwrite, `git reset --hard`,
   force-push, or `rm -rf` prior work autonomously. Ask, then wait.
3. **Additive over destructive.** New experiments write to NEW run dirs / NEW files /
   NEW configs. Do not mutate existing artifacts that back the v3 paper.
4. **Back up the data + models to the operator's Valar user area** so nothing is lost
   to a scratch purge (P0 in §7). Scope/target — Q16.
5. **Accumulate result numbers, never overwrite.** New measurements append as dated,
   reproducible rows in `AGENTS/V1/RESULTS.md`; prior numbers are kept (mark superseded,
   don't delete). [D24]

---

## Contents
- §0 goal of this doc · §1 where we are (incl. §1.5 root-cause, §1.6 infra, §1.7 results)
- §2 north star + constraints · §3 meeting threads · §4 experiment set
- §5 grilling Q&A (all resolved) · §6 decisions log (D1–D21) · §7 phased critical path

---

## 0. The three things this doc must make obvious

1. **Where we are** — what is already built and what the paper already claims (§1).
2. **Where we're going** — the north star after the 4.6.2026 meeting (§2–3).
3. **What we get at the end** — the concrete experiment set, each with
   description / motivation / dependency / ETA (§4), and the open decisions that
   gate them (§5).

---

## 1. Where we are (shipped) — [CONFIRMED from repo + paper]

### 1.1 The paper
Title: **"Can Vision Models Read? Probing Linguistic Structure via JEPA on Text
Screenshots."** Authors: Barış Cem Bakay, Halil İbrahim Kanpak (Koç Univ,
COMP547). 6-page ICML-style report. Central question: *do vision encoders acquire
linguistic structure when trained on rendered text?* Probe = retrieve the source
Wikipedia page from a 224×224 crop.

**Headline claims (current paper):**
- Image–text contrastive encoders dominate zero-shot: CLIP R@10 **0.779**, SigLIP
  **0.624**. All image-only encoders sit near random: DINOv2 0.058, ImageNet ViT
  0.031, random ViT 0.029, **I-JEPA 0.015 — below random**.
- **Platonic axis:** image-encoder↔text-encoder mutual-kNN alignment
  rank-correlates with retrieval R@10 at Spearman **ρ=0.83 (p=0.04, n=6)**.
- **Fine-tune:** DINOv2+SALAD on 50k pages → R@10 **0.915** (beats zero-shot CLIP).
  SALAD adds +9.3pp over an MLP head at matched backbone.
- **Layout-fingerprint failure mode:** painting out the top 15–30% of every test
  image drops trained Protocol-A R@10 by up to **0.29**, while *improving*
  cross-set Phase-2 R@1 by up to **6.7×**. Zero-shot CLIP/SigLIP move the *opposite*
  way (R@10 rises under blanking) → falsifies the "CLIP just OCRs the title" story.
- **I-JEPA + linear adapter to BERT[CLS]:** R@10 0.015 → **0.197** (13×) from a
  single 1280×768 linear map. Latent text alignment exists but is not zero-shot.
- **ColPali late interaction** does not rescue weak encoders (bottleneck = encoder,
  not aggregator). **LEACE:** title direction is ~rank-1 in CLIP/SigLIP but spread
  over a 16–64-dim subspace in fine-tuned DINOv2.

**The paper's own stated limitation (this is the crux for next steps):** at 224²,
*body text is sub-pixel and illegible* — so "reading" in the paper means
**heading-level text + layout only**, not fluent body content. Single seed. P=2000.

### 1.2 Codebase capabilities (`src/visword/`, `scripts/`, `slurm/`)
- **Backbones:** DINOv2 ViT-B/14, CLIP ViT-B/16, SigLIP ViT-B/16, ImageNet ViT,
  random ViT, I-JEPA ViT-H/14.
- **Heads/aggregators:** official SALAD (vendored, Sinkhorn-OT VLAD, 8448-d),
  CLS/MLP baseline.
- **Pretraining:** standard visual I-JEPA (`train_ijepa.py`) and **Text-Target
  cross-modal I-JEPA** (`train_ijepa_text.py`) — predicts frozen BERT reps of the
  page text from masked visual context. Trainable-block ablation (2/4/all-32).
  Full-res 490×490 DDP (`scratch/train_ijepa_*_ddp.py`).
- **Post-hoc adapters:** linear / MLP / bottleneck, image→BERT (`adapters.py`,
  `backbone_text_adapter_sweep.py`, `ijepa_text_adapter.py`, `probe_ijepa_text.py`).
- **Eval:** Protocol A (crop→page, leave-one-out gallery), Protocol B (anchor-pool
  cross-set), title-blanked variants.
- **Analyses:** Platonic alignment (mutual-kNN / CKA / Procrustes), LEACE causal
  erasure + rank sweep, ColPali late-interaction, attention↔text correlation.
- **Infra:** Valar SLURM cluster, slow-internet prefetch job pattern, conda env on
  `/scratch`, DDP (4× T4) with NCCL/barrier discipline, JSONL+PNG only (no trackers),
  one-run-dir-per-experiment provenance contract.

### 1.3 Data — [CONFIRMED]
- `Tevatron/wiki-ss-corpus` (~1.2M Wikipedia page screenshots; title + body text).
  Local cache = **376k pages, every image 980×980** (verified, §1.6).
- `hkanpak21/Wikipedia_SS_withanchors` — scroll-based anchor/positive/negative
  triplets for Protocol B (only 118 valid triplets usable).

### 1.4 Text-Target I-JEPA results so far (CONTEXT.md Session 4, 224², ~21k, 1 GPU)
- Text-Target All-Blocks: Protocol-A R@1 **0.79%**, Protocol-B R@1 **55.08%**,
  similarity gap flips negative→positive vs standard I-JEPA. Robust under 30% title
  blanking. **Caveat: text illegible at 224², tiny data, single GPU.** Full-res
  100k+ DDP reruns were "submitted, results pending."

---

## 1.5 Root cause of the "headline-only" problem — [CONFIRMED in code]

`NonOverlappingCropper` defaults: `crop_size=490`, `target_size=224`
([cropper.py:88](src/visword/data/cropper.py#L88), [default.yaml](configs/default.yaml)).
So every crop is a 490×490 native window **bilinear-downsampled 2.19× to 224**.
A Wikipedia body glyph (~12–16 native px) becomes ~6 px — sub-patch, sub-pixel,
illegible. **The legibility killer is the downsample ratio (crop_size/target_size),
not the 224 input itself.** Two consequences for the redesign:

- The cheapest possible fix is to crop at **native resolution** (`crop_size ≈
  target_size`, ratio ≈ 1). A native 224 window shows a few lines of legible body
  text — readable by *every* encoder including the 224-native CLIP/SigLIP.
- Native small crops cost ~N× more crops per tall page and **lose global page
  layout** (no crop knows it's the title vs a body paragraph). This is exactly why
  Barış wants a **global representation** → motivates a two-stream (global gist +
  local legible crops) design, independent of any autoencoder.

Knobs that are conflated today and must be separated in the redesign:
`crop_size` (page area covered) · `target_size` (px fed to ViT) · patch count ·
encoder native resolution (CLIP/SigLIP fixed at 224; DINOv2/I-JEPA flexible via
`interpolate_pos_encoding`). **Fair cross-encoder comparison constrains this — Q3/Q10.**

## 1.6 Infra reality (Valar — verified by SSH, read-only)
- Repo `/scratch/hkanpak21/VISWORD`; conda env `/scratch/hkanpak21/conda_envs/visword` (exists).
- **Data cache = 376,000 rows; every image is 980×980** (square top-of-page Wikipedia
  screenshots, not tall pages). Default cropper → 4× 490-quadrants → 224 (the 2.19×
  squash). **Native-224 crops on a 980² image = a legible ~4×4 grid (≈16 crops/page).**
- Anchors cache present: `images/`, `metadata.jsonl`, **`triplets_train.jsonl` +
  `triplets_val.jsonl`** (Protocol B has *train* triplets too, not just the 118 val).
- GPUs (partition `ai`, infinite wall): T4 16GB, **A40 48GB ×8**, V100 32GB, A6000;
  `avg`: **L40S 48GB**; `kutem_gpu`: A100. `t4_ai`: 8h wall. High-res/large-batch is
  feasible on A40/L40S/A100 — the T4-only assumption in the configs is outdated.
- **QOS ≈ 1 concurrent GPU job** → jobs run **sequentially**; plan resumable ≤8h
  chunks; the whole experiment set is serialized in wall-clock (ETAs reflect this).
  Login-node `nohup` disabled (use `setsid`).
- **Footprint & persistence:** `data/wiki_ss` = **125G**, anchors 1.7G, `runs/` = **20G**
  (all on `/scratch`, which can be purged). `/scratch` has 235T free; **`/home` has
  1.7 PB free** (persistent). HF cache = 133G in `/home/hkanpak21/.cache/huggingface`,
  already holding wiki-ss-corpus + anchors + BERT/distilBERT/Qwen — **but NOT
  Pix2Struct/Donut/Nougat/ColPali** (must be prefetched for offline compute nodes).
- **Operator = `hkanpak21`** (verified) — "my user part" is this account; data is
  already here. Gaps: persistent backup + new-model prefetch (→ P0).

## 1.7 What results actually exist (verified) — [FINDING → D18]
- 47 run dirs, **newest 2026-04-28** — they back the v3 paper (zero-shot grid,
  fine-tune grid, Platonic, LEACE, attention, adapter). Adapter baseline confirmed
  (`runs/ijepa_adapter`: I-JEPA→BERT no-adapter R@10 = 0.015).
- **Barış's June "large jepa pretraining submitted" produced NO run dirs.** No
  full-res Text-Target JEPA results exist on scratch — only the *code*. So "build on
  Barış's state" = build on his **code + the adapter baseline**, NOT on finished JEPA
  results. E4's first milestone is simply *a JEPA-reader run that completes* (efficient,
  A40/L40S, resumable) — exactly the "we can't even finish one model" failure to avoid.
- **GIT DIVERGENCE [FINDING → D23, preservation-critical].** The Valar working copy has
  **diverged from GitHub at `8035279`**. GitHub `origin/master` = + JEPA code
  (`acfc947`, `a1627a7`). Valar HEAD = + **three paper commits NOT on GitHub**
  (`fdd5dda` bibtex, `ad87b41` Tier-A hyperparams+reproducibility, `ff6b4bd` appendix:
  blank example + LEACE rank-sweep), **+ uncommitted changes** (modified
  `platonic_vs_retrieval.pdf`, new scripts, a deleted notebook). Valar HEAD appears
  **detached** (no current branch). → Barış's latest report lives only on Valar, off
  GitHub, unanchored. **Do NOT merge/push/reset without operator OK.** Reconcile safely
  in P0 (branch+commit on Valar to anchor it, then decide the merge).
- **Latest report (Valar, dated 2026-05-02): same numbers as v3**, plus a hyperparams
  table, an appendix, the "0.932 = 7-pool tie-break artifact" footnote, and a
  reproducibility line: **all experiments on ONE Tesla T4, ≈66 GPU-h, English Wikipedia
  only, seed 42.** Protocol-B anchor-pool = 118 val triplets **derived from
  wiki-ss-corpus itself** (anchor + same-page positives outside the anchor's crop
  neighbourhood + hard negatives from other pages), built once, not reused for training.

---

## 2. Where we're going — north star — [CONFIRMED 1.6.2026 grilling]

Course iteration (not yet an external venue). The operative realization from the
meeting: **the current setup only lets the model read headlines/layout. That must
be fixed so the model perceives ALL text in the image — every Wikipedia page's
body text, not just its title.**

**Central question (refined): _how do vision models read text, and can they read it
efficiently?_** "Efficiently" = without from-scratch pretraining or giant VLMs —
parameter-efficient fine-tuning of pretrained encoders at modest resolution.

Dual goal:

1. **Comparative study (keep + strengthen):** understand how different pretraining
   methods behave under the *all-text-legible* setting and what correlates with
   success (the Platonic / text-alignment axis). Families: image-text contrastive
   (CLIP, SigLIP), image-only SSL (DINOv2, I-JEPA), supervised, random — **plus a
   new family: document/text-pretrained vision encoders** (HF models already trained
   to read rendered text). This is the existing scientific spine, extended.
2. **Build a JEPA reader:** a JEPA model that perceives text from images, **by
   fine-tuning a pretrained backbone (never from scratch)**. Evaluated through the
   **retrieval task we already established** (Protocol A/B), extended for OCR + VQA / arXiv.

**Operational target for "perceive all text" [CONFIRMED → D5, refined D12]:**
content-level retrieval (body drives the match) **AND** alignment to text (the JEPA
reader predicts BERT-of-body; a *perfect-text upper bound* uses the ground-truth
`text` field — no OCR engine) **AND** an interpretability deliverable: **each model
must show where it attends** (attention maps over the page, correlated with text
regions). A feature→text decode probe (CER/WER) is *optional/lightweight*, not a
heavy build.

### 2.1 Hard constraints from grilling
- **C1 — No from-scratch pretraining.** We cannot finish even one such model in
  budget. Use **pretrained backbones, fine-tuned on our data**; prefer HF models
  **already fine-tuned to read text/documents** (Pix2Struct, Donut, ColPali, TrOCR,
  SigLIP2, Florence-2, Qwen2.5-VL vision tower, Nougat…). This kills the planned
  full-res-from-scratch 100k DDP I-JEPA runs. The "JEPA" work becomes
  *fine-tune a pretrained context encoder against a frozen BERT target*. **[D4]**
- **C2 — Text-aware cropping heuristics.** Today's grid cropper produces junk: empty
  crops, crops with only a fragment of a line, mid-line/mid-glyph cuts. Add heuristics
  so each crop is a *meaningful text unit*: drop empty/low-text crops, snap crop
  boundaries to inter-line whitespace (horizontal projection profile / light layout
  detection), avoid cutting lines. **[D6]**
- **C3 — Deliverables include OCR + attention**, not just retrieval R@k. **[D5]**

---

## 3. The five meeting threads (4.6.2026, Barış) — raw → interpreted

Meeting was in mixed TR/EN. Translation + my interpretation below; correct me where I'm wrong.

| # | Raw note | My interpretation | Status |
|---|---|---|---|
| T1 | "Redo all the training dataset." | Rebuild the training set — legible native-224 crops + text-aware cropping (not new domain, not from-scratch). | [DONE → D6/D19, E1] |
| T2 | "Visual question answering." | Add VQA as a downstream task to test *understanding*, not just retrieval. | [P3 → D15, E11] |
| T3 | "Understanding with arXiv papers." | Use arXiv papers as a harder reading domain beyond Wikipedia. | [P2 → D15, E10] |
| T4 | "JEPA pre-training." + "target encoder = BERT, take embeddings from BERT, then do retrieval with the JEPA model." | Make **Text-Target I-JEPA** (vision context encoder predicts frozen BERT *body* text) the *main* retrieval model — fine-tuned, not from scratch. | [DONE → D11/D13/D20, E4] |
| T5 | "Image reading: tokenizer solution; pixel space reads almost no text; 224×224 distorts resolution / loses text; need a GLOBAL representation; an autoencoder (CAE?) can address the HIGH-FREQUENCY nature of text." | Root cause = the 2.19× downsample (§1.5). Fixes: native-224 (E1) + global-page token (E6, in P1) + high-freq AE (E7, later arm). | [DONE → D19, E1/E6/E7] |

---

## 4. Candidate experiment set (DRAFT — gated by §5 answers)

> No-from-scratch (C1) means everything here is **fine-tune / inference scale**
> (hours–days on a few GPUs), not multi-week pretraining. No fixed deadline (D10);
> Valar QOS≈1 runs jobs **sequentially**, so the set is serialized in wall-clock.
> ETAs are rough per-job estimates.

| ID | Experiment | Description | Motivation | Depends | ETA |
|---|---|---|---|---|---|
| E1 | **Legible dataset rebuild + text-aware cropping** | Re-crop the 980² pages at **native 224** (`crop_size=224,target_size=224`, no downsample) → ~4×4 legible grid [D19]; add C2 heuristics: drop empty/low-text crops, snap crop bounds to inter-line whitespace (projection profile / light layout), reject fragment crops. New disjoint eval slice. | Precondition for every reading claim; user's cropping-heuristics ask. | — | 2–4 d |
| E2 | **Re-baseline the grid @ legible res** | Re-run 6-encoder Protocol-A/B + Platonic alignment at native res. Does the ordering survive legibility? Re-run the title-blanking test. | The comparative spine must be recomputed at the new res. | E1 | 2–3 d |
| E3 | **Add document/text-pretrained family** | Add Pix2Struct, Donut, Nougat, ColPali/ColQwen2 to the grid. Extract a comparable pooled embedding from each vision tower (ColPali stays multi-vector → late-interaction protocol). | C1 + Barış: likely a new top family → novel comparative result. | E2, Q10 | 3–5 d |
| E4 | **Text-Target JEPA reader — build on Barış's scaffolding** | Use his `models/ijepa_salad.py` + `ijepa_text_target_{mlp,salad}_30k` configs and the text-target checkpoint. **M0: get the checkpoint** (Barış shares it OR re-run pretrain under hkanpak21 — checkpoint is permission-denied, D25). **M1: fill his TBD head rows** (MLP/SALAD-30k retrieval). **M2: our delta** — switch target title→**body** (`text_source:text`, D9/D20), add **page-level re-identification** eval (D13), report params/throughput. | The JEPA reader contribution; "can a non-reader be taught to read efficiently?" Beat his adapter sweep. | P0(ckpt) | 4–7 d |
| E5 | **Perfect-text upper bound** | Ground-truth `text` → BERT → retrieve = the perfect-OCR ceiling vs the visual pipeline [D12]. (The feature→text *adapter* evidence is already done — Barış's image→text capacity sweep, A3/D25 — so no decode probe needed; our add is the ceiling row + a body-text version.) | D5/D12: bound the visual pipeline against perfect text. | E2 | 1 d |
| E6 | **Global-page representation (two-stream)** | Low-res whole-page gist token + legible local crops, fused into the page descriptor. | Barış: recover the global layout that small native crops discard. | E1 | 3–5 d |
| E7 | **High-freq AE front-end (ablation arm)** | Conv/wavelet AE to preserve glyph high-freq; compare vs plain native crops at matched patch budget. | Barış T5 novel arm; only pays off if patch-budget-constrained. | E2 | 4–6 d |
| E8 | **Attention "where it reads" viz** | Per-encoder attention maps + attention↔text-region correlation. Does the reader attend to body vs title? | D5/C3: models must show where they attend. | E2 | 1–2 d |
| E9 | **Confound control** | Random title/region masking during fine-tuning; verify the layout-fingerprint failure mode is gone at legible res. | The paper's universal pathology must not survive. | E1 | 2–3 d |
| E10 | **arXiv reading domain** | arXiv-paper screenshots; rerun retrieval + OCR probe + attention there. | Barış T3: harder real-reading test beyond Wikipedia. | E1, Q5 | 4–6 d |
| E11 | **VQA downstream** | Document VQA on the encoder/reader (dataset + role per Q6). | Barış T2: retrieval ≠ understanding. | E4, Q6 | 4–7 d |

**End state (what we get):** a legible-resolution Wikipedia (and arXiv) dataset with
meaningful text-aware crops; a re-baselined cross-family comparison (now incl. a
document/text-pretrained family) showing what pretraining correlates with *reading*;
a fine-tuned **Text-Target JEPA reader** evaluated on retrieval + an OCR-decode probe;
attention maps showing where each model reads; the layout confound controlled — all
written up as the next course-report iteration.

---

## 5. Open questions (grilling agenda) — answered inline as we go

> Resolved answers will be folded into §2–4 and re-tagged [DECIDED].

- **Q1 — Paper status/venue.** [DECIDED → D1] Course iteration.
- **Q2 — North-star thesis.** [DECIDED → D2] Comparative study + JEPA reader; read all text.
- **Q3 — Reading/resolution strategy.** [ACTIVE] Native-res crops (kill downsample)
  vs. higher-res large crops vs. learned tokenizer vs. high-freq AE + global token.
  Which is *most meaningful*? (User wants to discuss — my recommendation below.)
- **Q4 — Wiki dataset redo.** [ACTIVE, merged with Q3] What the rebuilt Wikipedia
  set looks like once "all text perceived" is the target (crop unit, scale, splits).
- **Q9 — Operational definition of "perceive all text."** [DECIDED → D5] Content
  retrieval + OCR-decodable + attention viz.
- **Q10 — Cross-encoder fairness.** [DECIDED → D16] Native-224 crops for all (primary);
  high-res rows for flexible encoders (secondary); report params+throughput.
- **Q11 — OCR role.** [DECIDED → D12] Ground-truth-text upper bound; no OCR engine.
- **Q12 — HF shortlist.** [DECIDED → D8] Pix2Struct, Donut, Nougat, ColPali/ColQwen2.
- **Q13 — JEPA reader base encoder.** [DECIDED → D11] SSL lead (I-JEPA + DINOv2),
  Pix2Struct/Donut ceiling.
- **Q14 — JEPA reader input unit.** [DECIDED → D13] Crop-based training, page-pooled
  retrieval; deployment unit = page.
- **Q15 — What is a "relevant doc"?** [DECIDED → D14] Same-page re-identification.
- **Q16 — Backup scope (P0a).** [DECIDED → D22] Additive `/home` copy of `data/` +
  `runs/` + `paper/` on Valar; originals untouched.

**All grilling questions Q1–Q16 are now resolved or scheduled.** Remaining genuinely
open items are deferred-by-design (arXiv/VQA dataset choices in P2/P3) or my-call
items flagged for your sign-off in §7.
- **Q5 — arXiv.** [DECIDED → D15] Phase 2, after Wikipedia. Source/dataset finalized then.
- **Q6 — VQA.** [DECIDED → D15] Phase 3, after arXiv.
- **Q7 — Text-Target JEPA design.** [DECIDED → D9/D11/D20] Frozen BERT target on the
  **body** text, token-level Smooth-L1; predictor + last blocks trainable; retrieval
  via page-pooled encoder features (SALAD not required for the JEPA reader).
- **Q8 — Compute + deadline.** [DECIDED → D10/§1.6] A40/L40S/A100 available; QOS≈1
  (serial execution); no fixed deadline.

---

## 6. Decisions log (filled during grilling)

- **D1 [DECIDED]** Deliverable = course iteration, not (yet) an external venue. (Q1)
- **D2 [DECIDED]** North star = (a) comparative study of pretraining methods under
  an all-text-legible setting + (b) a JEPA text-perceiving model, both evaluated via
  the established retrieval task. The model must read ALL text, not just headlines. (Q2)
- **D3 [FINDING]** The headline-only limitation is caused by the 2.19× crop→resize
  downsample (crop_size 490 → target 224), confirmed in code. Fixing the downsample
  is the precondition for everything. (§1.5)
- **D4 [DECIDED, refined]** No *random-init* training. **Continued-pretrain /
  fine-tune of pretrained backbones is the mode** (this is exactly what Barış's
  I-JEPA infra already does). Prefer HF text/document-pretrained encoders. Avoid the
  expensive end (full-res 90k all-blocks 48h DDP) as the *default*; keep an efficient
  fine-tune variant. Harvest Barış's submitted full-res results if/when they land. (C1)
- **D5 [DECIDED]** Success target = content retrieval **+** OCR-decodable features
  **+** attention "where it reads" visualization. (Q9)
- **D6 [DECIDED]** Add text-aware cropping heuristics (no empty crops, no fragment /
  mid-line cuts). (user follow-up)
- **D7 [DECIDED]** Comparative study gains a 5th family: document/text-pretrained
  vision encoders. (Q9 follow-up)
- **D8 [DECIDED]** Document family + fine-tune bases locked to: **Pix2Struct, Donut,
  Nougat, ColPali/ColQwen2.** (Not SigLIP2/Florence-2/Qwen for now.) (Q12)
- **D9 [DECIDED]** JEPA reader target = **full body-text BERT[CLS]** (not title).
  Removes the headline confound at the supervision source. (Q7)
- **D10 [DECIDED]** No fixed deadline → prioritize by scientific value; deliver a
  recommended critical path, not a calendar. (Q8)
- **D11 [REVISED after S1 + code/infra check]** JEPA reader **primary base = I-JEPA
  ViT-H/14** (the clean *within-base before/after* method claim: 0.015 → X; also what
  the existing code is wired for — 1280-d, patch-14, HF `interpolate_pos_encoding`).
  **Secondary = DINOv2 ViT-B/14** as a robustness + efficiency check, with the scale
  difference stated, not hidden. **Pix2Struct/Donut are NOT JEPA bases** — they're
  enc-dec/Swin and the code is /14-1280-specific; they stay **frozen ceilings in the
  comparison grid (E3)**. See D17 for why this is the honest comparability story.
- **D12 [DECIDED]** OCR = ground-truth `text` field as perfect-text upper bound; **no
  OCR engine build**. Feature→text decode probe optional/lightweight. (Q11)
- **D13 [DECIDED]** Deployment/eval unit = **page** ("give a page → retrieve relevant
  docs"). Page embedding = pooled legible-crop embeddings (or a doc encoder's
  whole-page embedding). Training stays crop-based (efficient, matches encoder
  limits); retrieval protocol extends to **page→retrieve**. (Q14)
- **D14 [DECIDED]** "Relevant doc" = **same-page re-identification** (page query;
  held-out view retrieves its own page from the gallery). No new relevance labels. (Q15)
- **D15 [DECIDED]** Strict phase order: **P1 Wikipedia → P2 arXiv → P3 VQA.** arXiv and
  VQA start only after the Wikipedia core gives nice results. **Build on Barış's master
  branch** (the I-JEPA infra); the paper is still v3 (full-res JEPA results unwritten).
  (Q5, Q6)
- **D16 [DECIDED by me, Q10]** Cross-encoder fairness: feed **native-resolution 224
  crops to every encoder** (the common denominator incl. 224-native CLIP/SigLIP) as
  the primary comparison; add **high-res rows** only for resolution-flexible encoders
  (DINOv2, I-JEPA, doc models) as a clearly-labeled secondary axis. Report
  trainable-params + throughput next to R@k so "efficiently" is measured, not asserted.
- **D17 [DECIDED — answers S1 "are they really comparable?"]** Three comparisons,
  three validity levels: **(A) method claim = within-base before/after** (same encoder,
  only the fine-tune changes) — rigorously fair, this is the core result; **(B) "which
  pretraining reads" grid** — across-arch/scale, NOT directly comparable on absolutes,
  so report params/throughput and read it as *family-level tendency* (the v3 paper
  already flags I-JEPA being 5× bigger yet worse); **(C) doc models** — different
  architecture class → presented as a labeled *ceiling/reference family*, never as
  matched competitors. We do NOT compare different encoders *as JEPA bases*.
- **D18 [CORRECTED 6.6]** Barış's full-res Text-Target JEPA **did finish** — checkpoint
  dated 2026-06-03 in **bbakay22's** scratch (not hkanpak21's — that's why §1.7 missed
  it); text-target zero-shot R@10 = 0.029. **But it is permission-denied to us** (D25).
- **D19 [DECIDED]** "Native-224" = `crop_size=224, target_size=224` (no downsample),
  NOT 490→224. Downsampling wrecks glyphs (your S2 note). High-res secondary =
  `target_size≥490` no-downsample, on A40/L40S.
- **D20 [FINDING]** `text_source` config already switches title↔body, and
  `train_ijepa_text.py` already predicts **token-level** BERT `last_hidden_state`
  (Smooth-L1, padding-masked) — richer than CLS. So D9 (body target) is a 1-line
  config change; open micro-choice: keep token-level vs switch to CLS; note BERT
  truncates body to `max_text_tokens` so "all text" is bounded by that cap.
- **D21 [DECIDED]** Preservation is a hard rule (see §⚠): never delete Barış's work or
  the paper; deletions of prior work require operator confirmation; all new work is
  additive; back up data/runs/paper to persistent `/home` (P0). Operator = `hkanpak21`.
- **D22 [DECIDED]** P0a backup scope = additive copy of `data/` + `runs/` + `paper/`
  from `/scratch` → `/home` on Valar; originals untouched, nothing mirrored off-cluster. (Q16)
- **D23 [FINDING, preservation-critical]** Valar repo diverged from GitHub at `8035279`
  (§1.7): Barış's 3 latest *paper* commits + uncommitted figure live only on Valar,
  on a likely-detached HEAD; GitHub has the JEPA code commits. Latest report = same
  numbers + appendix/hyperparams/reproducibility (single T4, ≈66 GPU-h). **Reconcile
  safely first (P0): anchor Barış's Valar commits on a branch + commit the figure
  BEFORE the backup; do not merge/push/reset without operator OK.**
- **D24 [DECIDED]** Accumulate result numbers, never overwrite — append dated
  reproducible rows in `AGENTS/V1/RESULTS.md`; keep superseded rows. (user; §⚠ rule 5)
- **D25 [MAJOR UPDATE 6.6 — Barış pushed `1bafedc`, we fast-forwarded to it].**
  Re-scopes the plan:
  - **JEPA reader is mid-flight, not greenfield.** Barış added `models/ijepa_salad.py`
    (I-JEPA + SALAD head loading the text-target checkpoint) + configs
    `ijepa_text_target_{mlp,salad}_30k`; head results are **TBD** (running). → E4
    *builds on these*, does not re-create them.
  - **Checkpoint blocker.** The pretrain checkpoint is in `bbakay22` scratch,
    **permission-denied** to us. → P0 adds: Barış shares it (shared dir / perms / weights)
    OR we re-run the pretrain under hkanpak21 from his configs.
  - **Resolution nuance.** His JEPA pretrain is full-res 490 (no downsample = legible);
    the 2.19× squash only afflicts the **comparison-grid default**. → E1/E2 legible
    rebuild targets the *grid*; the JEPA line is already legible. *Verify the 30k head
    configs don't silently inherit the default 490→224 downsample.*
  - **Our remaining deltas vs his title-target reader:** body-text target (D9),
    page-level re-identification (D13), document-pretrained family (E3), perfect-text
    bound (E5), attention (E8), confound control (E9), global token (E6).
  - **New numbers** logged in RESULTS.md (his image→text capacity sweep supersedes the
    single 0.197 adapter — both kept, D24).
  - **Divergence (D23) status:** `1bafedc` carries paper edits onto GitHub; whether the
    Valar May-2 paper commits are folded in or still separate needs a re-check.
- **D26 [DECIDED 6.6 — division of labor].** Barış *cannot* share the checkpoint (Valar
  perms), but he reads the repo. So:
  - **Barış's lane (delegated via tickets):** the **I-JEPA** Text-Target reader — his
    checkpoint, `ijepa_salad.py`, the `ijepa_text_target_{mlp,salad}_30k` heads, the TBD
    results. We do NOT touch this; tickets tagged `[BARIŞ]`.
  - **Our lane (non-overlapping, zero dependency on his checkpoint):** legible grid
    re-baseline + document-pretrained family + **our own reader on a base Barış isn't
    using — a pretrained HF masked autoencoder, MAE (`facebook/vit-mae-base/large`)** —
    body-text target + page-level eval + perfect-text bound + attention + confound
    control. The high-freq-AE arm (E7) folds into the MAE line. Bonus science:
    **I-JEPA (feature-prediction) vs MAE (pixel-reconstruction)** as a clean contrast.
- **GRILLING CLOSED.** Tickets written in `AGENTS/V1/issues/` (owner-tagged); then P0 → E1.

---

## 7. Critical path (phased) — the recommended order of operations

**P0 — Preservation & prefetch (before any experiment).**
0a. **Back up to persistent `/home`** (1.7 PB free): `data/wiki_ss` (125G), `wiki_ss_anchors`
   (1.7G), and `runs/` (20G) — protects the paper's results + dataset from a `/scratch`
   purge. Additive copy only; never move/delete the originals. Scope = Q16.
0b. **Prefetch new HF doc models** (Pix2Struct, Donut, Nougat, ColPali/ColQwen2) into the
   `/home` HF cache on the **login node** (compute nodes are offline). BERT already cached.
0c. Confirm the `visword` conda env + a 1-job smoke test still run green before scaling.
*(P0 is non-destructive and must precede E1. Honors the §⚠ hard rules.)*

**P1 — Wikipedia core (do this, in order). Gate to P2: "nice Wikipedia results."**

1. **E1** legible dataset rebuild + text-aware cropping (native res, no 2.19× squash;
   drop empty/fragment crops; snap to inter-line gaps). *Foundation — everything waits on it.*
2. **E2** re-baseline the 6-encoder grid + Platonic alignment + title-blanking at native res.
   *Does the headline-only ordering survive legibility? This is the first new result.*
3. **E3** add the document/text-pretrained family (Pix2Struct, Donut, Nougat, ColPali)
   to the grid. *Likely a new top family → the strengthened comparative result.*
4. **E4** Text-Target JEPA reader (evolve `train_ijepa_text.py`): SSL bases
   (I-JEPA + DINOv2) → predict **body-text BERT[CLS]**, param-efficient, page-pooled
   for **page→re-identification** retrieval; Pix2Struct/Donut as ceiling. *The method.*
5. **E5** perfect-text upper bound (ground-truth `text` → BERT → retrieve). *Bounds the visual pipeline.*
6. Run **E8** (attention "where it reads") and **E9** (confound control: random
   title-masking; verify the layout-fingerprint failure mode is gone) **across P1**, not after.

**P1 optional arms (Barış's ideas — need your go/no-go, §below):**
- **E6** global-page token / two-stream (recover layout lost by small native crops).
- **E7** high-frequency autoencoder front-end (preserve glyph detail; ablate vs plain
  native crops at matched patch budget). *Most novel, highest risk.*

**P2 — arXiv** (E10): only after P1 gate. Dataset/source decided then (prefer existing
arXiv-page data over building a renderer).

**P3 — VQA** (E11): only after P2.

**What P1 delivers (end state):** a legible Wikipedia dataset with meaningful crops;
a re-baselined cross-family comparison (now incl. document-pretrained models) answering
*how* different pretraining reads text under legibility; a parameter-efficient
Text-Target JEPA reader answering *can a vision model be taught to read efficiently*,
evaluated by page→page re-identification and bounded by perfect text; attention maps
showing *where* models read; and the title/layout confound demonstrably controlled —
written into the next iteration of the report, on top of Barış's master branch.

### Open for your sign-off (decisions I made or arms I flagged)
- **S1** — JEPA bases = I-JEPA + DINOv2 (SSL lead) + Pix2Struct ceiling (D11). OK?
- **S2** — Cross-encoder fairness via native-224-for-all + high-res secondary (D16). OK?
- **S3** — E6 (global token) and E7 (high-freq AE): in P1, or park as later arms?
- **S4** — Anything in §1 (current state) I have wrong, or any thread I mis-translated in §3?
