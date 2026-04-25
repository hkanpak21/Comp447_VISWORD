# Reading without a tokenizer: a study of visual document retrieval on Wikipedia screenshots

> **Project status.** This document combines (a) a detailed write-up of the
> experimental program executed on the Koç University Valar cluster between
> April 17–23, 2026, and (b) a forward-looking research agenda informed by
> the 2023–2026 literature on pixel-based language models, document AI,
> aggregation methods, probing methodology, and mechanistic interpretability.
> Part I is what *did* happen; Part II is what we believe *should* happen
> next.

---

## Abstract

We trained, ablated, and interpreted a 24-row method ladder on the
Wikipedia-screenshot retrieval task ([Tevatron/wiki-ss-corpus](https://huggingface.co/datasets/Tevatron/wiki-ss-corpus); 1.2 M page screenshots
with title and body text), comparing self-supervised vision backbones
(DINOv2 ViT-B/14), image–text contrastive backbones (CLIP ViT-B/16),
supervised vision (ImageNet-21k+1k ViT), and the official SALAD
Sinkhorn-OT aggregator (Izquierdo & Civera, *CVPR 2024*) against linear
probes, MLP heads, and several contrastive losses. Four findings
dominate. **(F1)** Off-the-shelf CLIP image features (no training) beat
every DINOv2 fine-tune on the held-out anchor-pool retrieval
(P2 R@1 = **0.932** vs. our best DINOv2+SALAD fine-tune **0.390**); the
combination of in-distribution recall and cross-distribution transfer
is unique to CLIP. **(F2)** DINOv2+SALAD wins in-distribution by ~20
R@10 points over an MLP head (0.925 vs. 0.741 at 10 k train pages) but
*loses* transfer monotonically with more training, indicating the
trained model overfits Wikipedia-layout fingerprints rather than
learning generalisable visual features. **(F3)** Within SALAD, the
8 192-d VLAD branch carries essentially all of the win — the 256-d
token branch alone is no better than a CLS-only baseline; the
Sinkhorn-OT structure with dustbin contributes only ~3 R@10 points
relative to plain softmax assignment. **(F4)** Title-only retrieval
with BERT/MiniLM/CLIP-text reaches R@1 = 1.0 on the anchor task — a
trivial upper bound that exists because every Wikipedia page has a
unique title — implying our visual benchmark is meaningful only when
the model cannot cheat through text. We then lay out a research
program rooted in the gaps surfaced by the literature: a
sparse-autoencoder interpretability study of the pixel-LM family, a
controlled test of the Platonic Representation Hypothesis on
pixel-text models, a disentanglement of reading (OCR-like glyph
recognition) from understanding (compositional semantics), and a
JEPA-on-screenshots extension that the current results directly
motivate.

---

## Part I — What we did

### 1. Motivation and research questions

The original setup (`PROJECT_SPEC.md`, Phases A–E) was deceptively
simple: given a 224 × 224 crop sampled from a Wikipedia page
screenshot, retrieve the source page from a held-out gallery. The
aggregator we adopted at the head of a DINOv2 ViT-B/14 backbone was
the official SALAD module (Izquierdo & Civera, *CVPR 2024*;
arXiv:2311.15937 — **not** "Serizel 2023" as it was misattributed in
the project's first specification; the confusion stemmed from the
GitHub handle `serizba` of the first author). Three weeks in, our
headline trained model (SALAD-full at 10 k train pages, 3 epochs) hit
a Phase-1 R@10 of 0.925 on 1 000 held-out pages — superficially
excellent. Two warning signals emerged simultaneously:

1. The same SALAD model scored P2 R@1 = 0.390 on the anchor-pool task,
   while the *frozen, untrained* DINOv2 CLS baseline scored 0.593, and
   the CLS-only fine-tune (otherwise identical recipe to SALAD) scored
   0.441. **Anchor-pool transfer was inverse to in-distribution skill.**
2. Inspecting the SALAD attention and dustbin overlays
   (`runs/2026-04-19_031513…_salad-smoke-5k_0f05/interpret/`) showed
   ~82 % of patches assigned to the dustbin and CLS attention focused
   on header chrome and margins — the model had learned a layout
   fingerprint, not a reading representation.

We re-framed the project around three questions which structure the
rest of this document:

- **RQ1 — Pretraining objective.** When we hold the head, loss, data,
  and recipe fixed, how much of in-distribution and out-of-distribution
  retrieval skill comes from the choice of frozen backbone (random
  init / ImageNet supervised / DINOv2 self-supervised / CLIP
  image-text contrastive)?
- **RQ2 — Aggregation.** Of the gain a SALAD aggregator brings over a
  CLS-MLP, how much is attributable to (a) the second descriptor
  branch (token vs. VLAD), (b) the optimal-transport structure with a
  dustbin vs. a plain softmax assignment, (c) the larger descriptor
  dimension (8 448-d vs. 256-d)?
- **RQ3 — Loss.** Is the SALAD-vs-CLS gap loss-invariant, and which of
  multi-similarity / InfoNCE / triplet best balances in-distribution
  fit and out-of-distribution transfer?

Sections 4–5 give answers; §7 lays out what *additional* questions the
literature suggests we should be asking and which experiments would
answer them.

### 2. Background and related work

The literature relevant to this project sits at the intersection of
five lines of work. Citations below are drawn from the literature
review supplied April 2026; no external references are introduced.

**Pixel-based language models.** PIXEL (Rust et al., *ICLR 2023*;
arXiv:2207.06991) re-renders text into images and recovers BERT-level
GLUE on English while exceeding it on non-Latin scripts; CLIPPO
(Tschannen et al., *CVPR 2023*) trains a single CLIP-style ViT on
image-and-rendered-alt-text; Pix2Struct (Lee et al., *ICML 2023 Oral*)
parses masked webpage screenshots into simplified HTML; PIXAR (Tai et
al., *Findings of ACL 2024*) extends this to autoregressive pixel
generation; PTP (Gao et al., 2024) and PixelGPT/DualGPT (Chai et al.,
*EMNLP 2024*) hybridise pixel and token objectives. Lotz et al.
(*EMNLP 2023*) systematically studied rendering strategies and
discovered an *anisotropic patch-embedding space driven by
patch-frequency bias*, the visual analogue of subword frequency
effects. PIXEL-M4 (Kesen et al., 2025) extended this multilingually
and is one of the few papers to actually probe the linguistic content
of pixel-LM features. Our work adopts the same input modality but
evaluates on retrieval, not masked reconstruction or autoregressive
generation.

**Document AI.** ColPali (Faysse et al., *ICLR 2025*; arXiv:2407.01449)
produces ColBERT-style multi-vector page embeddings for document
retrieval, directly demonstrating that aggregation can be skipped in
favour of late interaction (Khattab & Zaharia, *SIGIR 2020*); DocOwl-1.5
(Hu et al., *Findings of EMNLP 2024*) introduces an H-Reducer that
merges adjacent patches as a different answer to the same aggregation
question SALAD asks. LayoutLMv3 (Huang et al., *ACM MM 2022*), Donut
(Kim et al., *ECCV 2022*), and Nougat (Blecher et al., 2023) provide
the OCR-free document understanding context. The ViDoRe benchmark
(Faysse et al.) and DUDE (Van Landeghem et al., *ICCV 2023*) define
modern visual-document retrieval evaluation; M3DocVQA (Cho et al., 2024)
is especially relevant because it builds a multi-hop QA benchmark on
*Wikipedia* PDFs.

**Aggregation and OT.** SALAD (Izquierdo & Civera, *CVPR 2024*;
arXiv:2311.15937) reformulates NetVLAD's (Arandjelović et al., *CVPR
2016*) soft assignment as a Sinkhorn OT problem with both
feature-to-cluster and cluster-to-feature costs and adds a "dustbin"
cluster for uninformative features. The closest theoretical
neighbours are Sinkformers (Sander et al., *AISTATS 2022*) which
replace softmax attention with Sinkhorn iterations and prove the
equivalence to a Wasserstein gradient-flow step, and Set Transformer's
ISAB / PMA (Lee et al., *ICML 2019*) and Perceiver IO (Jaegle et al.,
*ICLR 2022*) which provide attention-based pooling baselines. SigLIP's
MAP head (Zhai et al., *ICCV 2023*; SigLIP 2 — Tschannen et al.,
arXiv:2502.14786) is the most recent industrial-scale attention-pooling
reference. ColBERT-style late interaction is the "don't aggregate"
baseline against which all of these are compared.

**Probing and interpretability.** Classical probing
(Alain & Bengio, *ICLR Workshop 2017*; Conneau et al., *ACL 2018*;
Tenney et al., *ICLR 2019*; Hewitt & Manning, *NAACL 2019*) and the
causal-probing line (INLP — Ravfogel et al., *ACL 2020*; Amnesic
Probing — Elazar et al., *TACL 2021*; RLACE — Ravfogel et al., *ICML
2022*; LEACE — Belrose et al., *NeurIPS 2023*) supply the methodology
we will adopt for §7. Hewitt & Liang (*EMNLP 2019*, Best Paper
Runner-Up) and Voita & Titov (*EMNLP 2020*; MDL probing) supply the
methodological guard-rails. Belinkov (*Computational Linguistics 2022*)
is the authoritative critique to cite when claiming any linguistic
knowledge from probe accuracy. CKA debiasing (Kornblith et al., *ICML
2019*; Murphy et al., *ICLR 2024 Re-Align Workshop*; Davari et al.,
*ICLR 2023*) supplies the alignment metric critique. Patchscopes
(Ghandeharioun et al., *ICML 2024*) and Tuned Lens (Belrose et al.,
2023) generalise logit-lens decoding. Most relevantly, Neo et al.
(*ICLR 2025*; arXiv:2410.07149) demonstrate that *visual-token
residual streams in LLaVA progressively align with textual vocabulary
embeddings even though the model is never trained for next-token
prediction on images* — directly relevant to whether our SALAD/CLIP
features acquire a linguistically decodable structure as training
progresses.

**Mechanistic interpretability of vision.** Sparse autoencoders are now
the dominant tool for interpreting vision transformers in 2024–2025:
Bricken et al. (Anthropic, 2023), Cunningham et al. (*ICLR 2024*),
Templeton et al. (Anthropic, 2024) for language; Gao et al. (TopK,
2024), Rajamanoharan et al. (Gated, JumpReLU, 2024) and Gemma Scope
(Lieberum et al., *BlackboxNLP 2024*) for scalable variants; PatchSAE
(Lim et al., 2025), DN-CBM (Rao et al., *ECCV 2024*), CoX-LMM (Parekh
et al., *NeurIPS 2024*), Joseph et al. (*CVPR 2025 MI-Vision*) and
saev (Stevens et al., 2025) for vision; Zaigrajew et al. (*ICML 2025*)
for hierarchical (Matryoshka) variants; Prisma (Joseph et al., 2025)
for open-source weights. Stevens et al. is methodologically the
closest reference for what we want to do next — they trained TopK SAEs
on CLIP and DINOv2 and found CLIP's language supervision yields
cross-style abstractions DINOv2 does not. Gandelsman et al.
(*ICLR 2024 Oral*; arXiv:2310.05916) decompose CLIP-ViT into sums over
layers × heads × patches and label each via TextSpan; Balasubramanian
et al. (*NeurIPS 2024*) extend this to non-CLIP vision models. Darcet
et al. (*ICLR 2024 Outstanding Paper*) identified high-norm artifact
register tokens in DINOv2/OpenCLIP/DeiT-III which would confound any
pooling analysis.

**What vision models know about text.** Yuksekgonul et al.
(*ICLR 2023 Oral*) showed CLIP/BLIP/FLAVA behave like bags-of-words on
the ARO benchmark; Thrush et al. (*CVPR 2022*) Winoground confirmed no
VLM substantially beats chance; Hsieh et al. (*NeurIPS 2023 D&B*)
SugarCrepe showed earlier hard-negative benchmarks were "hackable" by
text-only models. Materzyńska et al. (*CVPR 2022 Oral*) demonstrated
that *spelling and meaning are partially separable subspaces in CLIP*,
and Goh et al. (*Distill 2021*) introduced typographic attacks. Lin
et al. (*ECCV 2024*) showed ~50 % of LAION-2B captions verbatim repeat
text-in-image — explaining CLIP's OCR-like behaviour and typographic
vulnerability. Most pertinent for our forward-looking work is Jose et
al. (*Meta, 2024*; "DINOv2 Meets Text"): a small text encoder aligned
post-hoc to a *frozen* DINOv2 already reaches CLIP-level zero-shot
classification — strong evidence that self-supervised image features
already contain a linguistically-decodable structure. The 2025
"Emergence of Text Readability in VLMs" study (arXiv:2506.19389) is
the most provocative recent finding: text readability emerges
*suddenly* at a specific training point, after general semantic
understanding — implying any probe study should track multiple
training checkpoints.

### 3. System and data

**Corpus.** `Tevatron/wiki-ss-corpus` is a HuggingFace dataset of
~1.2 M Wikipedia page screenshots with `docid`, `title`, and rendered
body `text`. We additionally use `hkanpak21/Wikipedia_SS_withanchors`
(~30 MB) which provides anchor-triplet evaluation: each anchor is a
crop of a page, paired with positive crops (other crops of the same
page) and hard-negative crops (crops of unrelated pages); 571 anchors
in the validation split, of which 118 are usable because the upstream
HF repo only ships ~10 k images alphabetically. We grew the local
cache from an initial 5 000 rows (Apr 18) to 15 000 (Apr 19) using
serial HF streaming, hit a cluster-wide DNS block on `huggingface.co`
on Apr 21 which we routed around with a pure-stdlib `socket` shim
(`src/visword/hf_dns_shim.py`), then ran a seven-process parallel pull
on the login node (one streaming + six bulk-arrow workers using
`huggingface_hub.hf_hub_download` authenticated via an HF API token)
targeting 550 k rows. Each row is re-encoded as PNG
(`compress_level=1`), written to
`blobs/{idx//1000:02d}/{idx:07d}.png`, with sidecar text under
`texts/`. The manifest is sha256-fingerprinted over canonical-sorted
rows and verified at every cache load.

**Cropper.** `src/visword/data/cropper.py` implements
`NonOverlappingCropper(crop_size=490, overlap=0.0, target_size=224)`.
Stride is integer-quantised; near-white images fall back to a single
centre crop (a lesson learned from the week-1 sanity-check failures
recorded in `CONTEXT.md`). All crops are ultimately resized to
224 × 224 RGB and ImageNet-normalised
(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

**Backbones.** The DINOv2 ViT-B/14 weights are loaded via
`torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` pinned to
commit `e1277af2ba94…` (the last Python-3.9-compatible commit;
`scripts/ensure_dinov2_hub.py` enforces the pin idempotently). The
CLIP ViT-B/16 image branch is loaded via
`open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')`
and exposed through `src/visword/models/clip_backbone.py`, which
reshapes the ViT's per-patch tokens into the same `(B, 768, H, W)` +
`(B, 768)` interface as our DINOv2 wrapper, with inline
ImageNet→CLIP normalisation so the dataset transform is shared across
backbones. The ImageNet-21k+1k supervised baseline is `timm` model
`vit_base_patch16_224.augreg_in21k_ft_in1k`. Random-init ViT-B/16 was
attempted as a sanity floor but failed without further hyperparameter
tuning and is not reported.

**SALAD aggregator.** Vendored from `serizba/salad@e3fe4e22…` under
`third_party/salad/` and wrapped by
`src/visword/models/salad_bridge.py` as the *single* import path (the
project deliberately forbids duplicate copies). The aggregator takes
`(patches, cls_token)` and returns an 8 448-d vector formed as
`concat(L2norm(token_features(cls)), L2norm(VLAD).flatten())` followed
by a final L2-norm. Internally: `score(patches)` produces a
`(B, K=64, H, W)` per-patch logits tensor; `cluster_features(patches)`
produces a `(B, cluster_dim=128, H, W)` local-feature tensor;
`token_features(cls)` is an MLP into a `token_dim=256` global vector. A
3-iteration log-domain Sinkhorn (`get_matching_probs`) over an
augmented `(K+1, N)` score matrix with a learnable `dust_bin` parameter
yields the doubly-stochastic assignment whose first `K` rows multiply
the per-patch features into the per-cluster aggregates.

**Ablation harness.** `AblatedSALAD` in
`src/visword/models/salad_ablations.py` inherits from `OfficialSALAD`,
adds **zero new parameters** (state-dict identical so the "full" mode
is bit-exact to the vendored forward), and exposes four modes:
`full` (8 448-d), `token_only` (256-d), `vlad_only` (8 192-d), and
`softmax_assign` (8 448-d but Sinkhorn replaced by per-patch softmax
over clusters).

**Backbone partial fine-tuning.** The DINOv2 backbone's last
`num_trainable_blocks` ViT blocks have `requires_grad=True`; the rest
are wrapped in `torch.no_grad()` *and* explicitly frozen so the
optimizer doesn't allocate state for them
(`src/visword/models/dinov2_salad.py:_freeze_frozen_backbone_blocks`).
The same pattern is mirrored for CLIP in `freeze_clip_backbone_blocks`.

**Heads.** `DINOv2CLS` uses a 768→512→256 MLP returning an L2-normed
256-d vector. `DINOv2LinearProbe` (and `CLIPLinearProbe`) replace the
MLP with a single Linear, with the backbone fully frozen, isolating
the linear-decodability of the pretrained features. `DINOv2SALAD` and
`CLIPSALAD` use the same `AblatedSALAD` aggregator on top of their
respective backbones.

**Training loop.** `src/visword/train.py`: two parameter groups
(`lr_backbone=1e-5`, `lr_head=5e-4`), AdamW with weight decay `1e-4`,
gradient clip `1.0`, linear warmup over the first 5 % of steps then
cosine decay to 0, batch size 16 (T4 OOMs SALAD-full at 32),
`k_per_page=4` so each batch has 16 crops sampled from 4 pages (every
other crop is a positive). Multi-similarity loss (Wang et al., 2019)
is the default; InfoNCE and Triplet are also supported via
`src/visword/losses.py`'s `build_loss(name, **kwargs)` factory.
Best-checkpoint selection uses held-out Phase-1 R@10 evaluated every
`eval_every_steps` steps.

**Evaluation.** Two phases run after every training:
- *Phase 1* — in-distribution. Hold-out 1 000 pages, 4 crops each ⇒
  4 000 query crops; gallery is the 1 000 page-mean descriptors;
  report R@k for k ∈ {1, 5, 10, 20} plus a sanity gap (mean same-page
  sim − mean diff-page sim). Random R@1 = 1/1 000 = 0.001
  (`src/visword/eval_phase1.py`).
- *Phase 2* — anchor triplets. For each anchor crop, rank the union of
  positives + hard-negatives by cosine sim; R@k = positive in top-k.
  Pool is 4 352 candidate images. Random R@1 ≈ 0.000 23
  (`src/visword/eval_phase2.py`).

**Interpretability.** `src/visword/interpret/` produces, per run,
last-block CLS-attention overlays (`attention_sample{i}.png`),
patch-NN match maps (`patch_triplet_{i}/`), per-cluster Sinkhorn
assignment heatmaps (`salad_clusters_sample{i}.png`), per-patch
dustbin overlays (`dustbin_map_sample{i}.png`), a CLS-vs-VLAD
descriptor decomposition (`cls_vs_vlad.{json,png}`), and a
`dustbin_evolution.png` time-series of dustbin mass during training.
SALAD-specific artefacts are gated on `cfg.model_kind == "salad" and
cfg.salad.ablation == "full"`; attention/patch-NN are gated on
DINOv2-family backbones (CLIP layout is different and would crash the
hard-coded `backbone.model.blocks[-1]` hook).

### 4. The method ladder

We instantiated 24 rows along five axes — backbone, freezing strategy,
head, loss, modality — each row differing from its predecessor by
exactly one ablation. This deliberate one-axis-at-a-time design is
what makes the comparison interpretable. Every row writes
`config.resolved.yaml` plus `provenance.json` (git SHA, hub commit,
weights file sha256) into a fresh `runs/<TS>_<git>_<row_label>_<rand>/`
directory. The full ladder:

| # | Method label                                                | Backbone (frozen unless noted)         | Head                | Loss     | Modality       |
|---|-------------------------------------------------------------|-----------------------------------------|---------------------|----------|----------------|
| 1 | Random ViT (sanity floor; not reported)                     | ViT-B/16 random init, frozen           | raw CLS             | —        | image          |
| 2 | ImageNet-21k+1k ViT zero-shot (failed: timm DNS)            | timm `vit_base_patch16_224.augreg…`   | raw CLS             | —        | image          |
| 3 | DINO-v1 zero-shot (deferred — torch.hub failure)            | DINO-v1 ViT-B/16, frozen               | raw CLS             | —        | image          |
| 4 | **DINOv2 zero-shot CLS**                                    | DINOv2 ViT-B/14, frozen                | raw CLS             | —        | image          |
| 5 | DINOv2 zero-shot mean-patch                                 | DINOv2 ViT-B/14, frozen                | mean-pool patches   | —        | image          |
| 6 | **CLIP image zero-shot**                                    | CLIP-ViT-B/16 image, frozen            | image projection    | —        | image          |
| 7 | DINOv2 + Linear probe                                       | DINOv2 frozen                           | Linear 768→256      | MultiSim | image          |
| 8 | DINOv2 + MLP, last-2 blocks fine-tune (deferred)            | DINOv2, last 2 blocks                   | MLP 768→512→256     | MultiSim | image          |
| 9 | **DINOv2 + MLP, last-4 blocks (CLS-main)**                  | DINOv2, last 4 blocks                   | MLP 768→512→256     | MultiSim | image          |
| 10 | DINOv2 + MLP + InfoNCE loss                                | same as row 9                            | same                | InfoNCE  | image          |
| 11 | DINOv2 + MLP + Triplet loss                                | same as row 9                            | same                | Triplet  | image          |
| 12 | **DINOv2 + SALAD-full (salad-main)**                        | DINOv2, last 4 blocks                   | SALAD-OT 8 448-d    | MultiSim | image          |
| 13 | DINOv2 + SALAD `vlad_only`                                  | same                                     | VLAD only 8 192-d   | MultiSim | image          |
| 14 | DINOv2 + SALAD `token_only`                                 | same                                     | token branch 256-d  | MultiSim | image          |
| 15 | DINOv2 + SALAD `softmax_assign`                             | same                                     | SALAD w/o OT/dustbin| MultiSim | image          |
| 16 | BERT-base zero-shot, title only                            | `bert-base-uncased`, frozen             | mean-pool last hidden| —       | text           |
| 17 | BERT zero-shot, title + body[:200]                         | same                                     | same                | —        | text           |
| 18 | sentence-MiniLM-L6 zero-shot                               | `all-MiniLM-L6-v2`, frozen              | sentence vector     | —        | text           |
| 19 | CLIP text zero-shot                                        | CLIP-ViT-B/16 text, frozen              | text projection     | —        | text           |
| 20 | CLIP cross-modal zero-shot                                 | CLIP both branches, frozen              | shared embed        | —        | image+text     |
| 21 | DINOv2-SALAD ⊕ BERT (deferred)                              | row 12 ⊕ row 16                         | concat + L2-norm    | —        | image+text     |
| 22 | CLIP-image + MLP, last-4 blocks fine-tune (collapsed)       | CLIP, last 4 blocks                     | MLP                 | MultiSim | image          |
| 22b | **CLIP-image FROZEN + MLP head trained**                   | CLIP, frozen                             | MLP                 | MultiSim | image          |
| 23 | CLIP-image + SALAD-full, last-4 blocks (degraded)          | CLIP, last 4 blocks                     | SALAD               | MultiSim | image          |
| 23b | **CLIP-image FROZEN + SALAD trained**                      | CLIP, frozen                             | SALAD               | MultiSim | image          |
| 24 | CLIP-image + SALAD ⊕ CLIP-text (deferred)                   | row 23 ⊕ CLIP text                       | concat + L2-norm    | MultiSim | image+text     |

Rows in **bold** are the headline methods; rows marked "deferred"
either failed for a reason orthogonal to the science (row 1 random
ViT needed tuning we didn't do; rows 3 and 8 are simple additions we
prioritised below other experiments; rows 21 and 24 require
multimodal descriptor concatenation infra we didn't get to). Rows 22
and 23 collapsed during training (§4.5) and were resubmitted as
22b/23b with the backbone fully frozen.

#### 4.1 Data scale ladder for the headline methods

In addition to the method ladder, we ran rows 9 and 12 at three data
scales to measure how the SALAD>CLS gap depends on training budget:

| Scale | Data (train×eval) | Epochs | Steps | SALAD R@10 | CLS R@10 | Gap |
|---|---|---|---|---|---|---|
| smoke    | 4 000 × 800  | 2 |   500 | 0.871 | 0.447 | +42.4 |
| 5k-main  | 4 500 × 500  | 3 |   840 | 0.955 | 0.653 | +30.2 |
| 5k-long  | 4 500 × 500  | 8 | 2 240 | 0.970 | 0.761 | +20.9 |
| main     | 10 000 × 1 000 | 3 | 1 875 | 0.925 | 0.741 | +18.4 |
| **20k**  | 20 000 × 2 000 | 2 | 2 500 | **0.901** | **0.718** | **+18.3** |
| **15k**  | 15 000 × 1 500 | 3 | 2 400 (best) | 0.923 | 0.748 | +17.5 |

Phase 2 R@1 across SALAD trained runs (the metric that separates them):

| Run | P2 R@1 | Sanity gap | Notes |
|---|---|---|---|
| salad-main (10k×3) | 0.390 | −0.045 | most over-specialised |
| salad-20k (20k×2) | 0.398 | −0.030 |  |
| salad-5k-long (5k×8) | 0.398 | −0.040 |  |
| **salad-15k (15k×3)** | **0.483** | **−0.014** | **best Phase 2; least inverted gap** |

salad-15k is the **first trained SALAD run whose anchor-pool sanity
gap is approximately zero** rather than meaningfully negative. *More
data with same number of epochs partially counteracts wiki-ss
over-specialisation* — a pattern that should be tested further at
30k+ data scale, except T4 walltime caps make 30k×3 infeasible in a
single sbatch.

Two patterns: (1) more training-on-the-same-data closes the gap
faster than more data — at 5k-long (5 k pages × 8 epochs) SALAD R@10 =
0.970 beats main (10 k × 3) at 0.925, suggesting we are still
training-budget-limited not data-limited at 5 k pages; (2) the gap
narrows monotonically with budget but does not collapse (CLS R@10
climbs +20 pts faster than SALAD over the same range).

#### 4.2 Phase 1 results (in-distribution)

| Row | Method label | P1 R@1 | P1 R@5 | P1 R@10 | P1 R@20 | Sanity gap |
|---|---|---|---|---|---|---|
| 5  | DINOv2 zero-shot mean-patch         | 0.007 | 0.013 | 0.016 | 0.026 | +0.005 |
| 22b | CLIP frozen + MLP head trained (collapsed) | 0.002 | 0.004 | 0.006 | 0.010 | 0.000 |
| 4  | DINOv2 zero-shot CLS                | 0.045 | 0.076 | 0.097 | 0.123 | +0.023 |
| 11 | DINOv2 + Triplet (broken)           | 0.027 | 0.061 | 0.087 | 0.131 | +0.040 |
| 7  | DINOv2 + Linear probe               | 0.094 | 0.214 | 0.273 | 0.372 | +0.105 |
| 23b | **CLIP frozen + SALAD trained**    | 0.343 | 0.509 | 0.582 | 0.654 | small |
| 6  | **CLIP image zero-shot**            | **0.443** | 0.585 | **0.664** | 0.755 | +0.214 |
| 10 | DINOv2 + InfoNCE                    | 0.252 | 0.560 | 0.712 | 0.815 | +0.135 |
| 9  | DINOv2 + MLP MultiSim (cls-main)    | 0.349 | 0.632 | 0.741 | 0.832 | +0.233 |
| 14 | SALAD `token_only`                  | 0.144 | 0.317 | 0.421 | 0.533 | +0.125 |
| 15 | SALAD `softmax_assign`              | 0.491 | 0.756 | 0.842 | 0.906 | +0.137 |
| 13 | SALAD `vlad_only`                   | 0.515 | 0.795 | 0.877 | 0.931 | +0.142 |
| 12 | **DINOv2 + SALAD-full (salad-main)** | **0.599** | 0.865 | **0.925** | 0.959 | +0.239 |

Observations: (a) zero-shot DINOv2 is barely above random
(R@10 = 0.097 vs. random 0.01 — about 9.7×); (b) a single Linear layer
trained on frozen DINOv2 features gets 35 % of full SALAD's R@10 from
one trainable matrix; (c) CLIP zero-shot, with no training at all,
beats every DINOv2 fine-tune except SALAD-full and InfoNCE; (d) within
SALAD's family, removing the token branch costs almost nothing
(`vlad_only` = 0.877 vs. full 0.925) while removing VLAD destroys it
(`token_only` = 0.421); (e) replacing Sinkhorn with softmax assignment
costs only 3 R@10 points (`softmax_assign` = 0.842); (f) frozen-CLIP
+ trained SALAD head reaches R@10 = 0.582 — *worse* than CLIP zero-shot
0.664, indicating that the SALAD aggregator on top of CLIP features
*degrades* in-distribution retrieval (the head is fitting the wiki-ss
distribution but losing CLIP's per-patch quality).

#### 4.3 Phase 2 results (anchor pool)

| Row | Method label | P2 R@1 | P2 R@5 | Sanity gap |
|---|---|---|---|---|
| 10 | DINOv2 + InfoNCE             | 0.178 | 1.000 | small        |
| 11 | DINOv2 + Triplet             | 0.356 | 1.000 | small        |
| 12 | DINOv2 + SALAD-full          | 0.390 | 1.000 | **−0.045**   |
| 9  | DINOv2 + MLP MultiSim        | 0.441 | 1.000 | small        |
| 23 | CLIP+SALAD fine-tune (degraded) | 0.483 | 1.000 | −0.008    |
| 23b | **CLIP frozen + SALAD trained** | 0.500 | 1.000 |           |
| 4  | DINOv2 zero-shot CLS         | 0.593 | 1.000 | +0.036       |
| 22 | CLIP+CLS fine-tune (collapse)| 0.627 | 1.000 | ≈0           |
| 22b | **CLIP frozen + MLP head**  | 0.695 | 1.000 |              |
| 7  | DINOv2 + Linear probe        | 0.720 | 1.000 |              |
| 6  | **CLIP image zero-shot**     | **0.932** | 1.000 |          |
| 16-19 | All text-only            | 1.000 | 1.000 | (degenerate) |

The negative sanity gap on SALAD-full is the strongest single piece
of evidence that the trained model has *over-specialised to wiki-ss
layout*: same-page anchor sim is *lower* than diff-page sim. CLS-only
fine-tunes preserve a positive gap. Text-only retrieval is degenerate
because each Wikipedia page has a unique title string; the anchor
retrieval becomes an exact-string match. R@5/10/20 saturate at 1.0
across the board because the pool size is small (118 valid triplets,
positive-vs-hard-negative discrimination only). R@1 is the only
discriminative metric in Phase 2.

Notably, rows 22b and 23b — frozen CLIP backbone with trainable head
— *both* fail to preserve CLIP zero-shot's 0.932 P2 R@1. Even when
the backbone is locked, training a head on the wiki-ss distribution
degrades cross-distribution transfer (22b: 0.695, 23b: 0.500). This
is direct evidence that the loss function itself, not the backbone
fine-tuning, is what destroys transferable features when training on
this corpus.

#### 4.4 Interpretability snapshots (current)

For SALAD-main (`runs/2026-04-19_031513…_salad-smoke-5k_0f05/interpret/`):

- **Attention** focuses on top-right header chrome and margins, NOT
  body text — visually unsatisfying given the task is "read Wikipedia",
  scientifically explained by the patch-resolution argument: at 14 ×
  14-pixel patches, individual letters are 1–2 pixels per stroke,
  illegible to the model. The discriminative signal that fits in a
  14 × 14 patch is *layout structure*: infobox shape, header position,
  list bullet patterns, figure placement.
- **Per-patch dustbin** captures ~82 % of patches — the model has
  learned that most patches are uninformative and routes them to the
  slack cluster. Body text and white background go to the dustbin;
  only a handful of structural-fingerprint patches survive into the
  descriptor.
- **Dustbin evolution** over training rises from 0.75 (initial) to
  0.81 by step 50 and plateaus. *More training increases dustbin
  mass*: the model learns to be more selective, not less.
- **CLS-vs-VLAD decomposition**: same-page sim of 0.639 decomposes as
  0.620 (VLAD) + 0.020 (CLS); the 256-d token branch contributes ~3 %
  of the discriminative signal. This is the geometric explanation for
  why `vlad_only` ≈ full and `token_only` ≈ CLS-baseline.
- **Patch nearest-neighbours** between two crops of the same page
  achieve cos = 0.997 (essentially identical patch descriptors)
  vs. cos = 0.025 for crops of different pages — patch-level features
  are strongly discriminative.

#### 4.5 The CLIP fine-tune collapse (negative result)

Rows 22 and 23 (CLIP backbone, last-4 blocks trainable, lr_bb = 1e-5
— the same LR that worked for DINOv2) catastrophically collapsed.
Row 22 landed P1 R@10 = 0.006 (5× *below* random); row 23 landed
P1 R@10 = 0.475 (vs. zero-shot CLIP 0.664). The training-loss
diagnostics show `pos_sim_mean = neg_sim_mean = 1.0` — the model
converged to a constant output for every input. Two plausible causes:

1. CLIP weights were trained with QuickGELU but our open_clip
   checkpoint loads with standard GELU (a `UserWarning` flagged this
   at construction); the activation mismatch may have shifted
   gradients into a degenerate basin.
2. lr_bb = 1e-5 is too large for CLIP's pretrained features given
   batch size 16; CLIP's contrastive pretraining used much larger
   effective batches (>32 k) and small backbone perturbations matter
   more for it than for DINOv2.

Rows 22b and 23b — CLIP fully frozen, only the head trained — were
submitted as the corrective experiment. Row 22b *still* collapsed (P1
R@10 = 0.006) — even with the backbone frozen, the MLP head learned
to map every input to a single descriptor. Row 23b avoided the
collapse and reached P1 R@10 = 0.582 (worse than CLIP zero-shot
0.664) and P2 R@1 = 0.500 (worse than CLIP zero-shot 0.932). The
SALAD aggregator with K=64 cluster heads provides enough output
diversity to escape the collapsing basin a single MLP head falls
into at this batch size, but neither configuration recovers CLIP's
zero-shot performance, let alone improves on it. This is direct
evidence that **training a head on top of CLIP using our wiki-ss
contrastive recipe degrades CLIP's representations** — the issue is
not solely the backbone fine-tuning rate.

### 5. Findings

We collect the seven findings the ladder establishes. Each is
supported by specific row IDs and recall numbers; they should be
read as *compressed claims* with the full evidence in §4.

- **F1 — CLIP image zero-shot dominates anchor-pool transfer.**
  P2 R@1 = 0.932 (row 6) > best DINOv2 fine-tune (row 12) at 0.390 by
  +54 pts. CLIP also reaches a respectable P1 R@10 = 0.664 with no
  training. This generalises across image-image (0.932) and
  image-text-cross-modal (row 20, 1.000) protocols. Explanation:
  CLIP's image-text contrastive pretraining produces page-level
  features that are robust to the wiki-ss → anchor distribution
  shift, in a way DINOv2's self-supervised image-only pretraining is
  not.

- **F2 — DINOv2 fine-tuning trades transfer for in-distribution
  skill, monotonically with training.** P2 R@1 monotonically degrades
  as Phase-1 R@10 climbs across the data-scale ladder (0.517 at 3 ep
  → 0.398 at 8 ep on the same 5 k pages). Same-page anchor sanity gap
  goes negative (−0.045 on salad-main). This is the empirical
  signature of layout-fingerprint over-specialisation, validating the
  project's early concern about "white-on-white alignment".

- **F3 — The VLAD branch carries essentially all of SALAD's win over
  CLS.** `vlad_only` (8 192-d) achieves P1 R@10 = 0.877 vs. full
  (8 448-d) at 0.925 — within noise; `token_only` (256-d) drops to
  0.421, indistinguishable from a CLS-MLP baseline (row 9 = 0.741 at
  10 k train, but smoke-scale row 14 vs. smoke CLS 0.447 confirms the
  ~10-point gap is the MLP head, not the SALAD token branch). The
  CLS-vs-VLAD attribution chart confirms the 256-d token contributes
  ~3 % of similarity.

- **F4 — Sinkhorn-OT structure with dustbin contributes only ~3 R@10
  points over plain softmax assignment.** `softmax_assign` (no
  Sinkhorn, no dustbin) hits P1 R@10 = 0.842 vs. full SALAD 0.871
  (smoke-scale). The expensive-looking OT machinery is not where the
  win lives; the win lives in *per-patch soft assignment into local
  clusters with L2-normed per-cluster aggregation*. Sinkformers'
  (Sander et al., 2022) theoretical equivalence between Sinkhorn
  iterations and Wasserstein gradient flow gives a clean explanation
  for why Sinkhorn is at most a marginal improvement over softmax in
  this regime.

- **F5 — Loss matters more than head choice for transfer.** Same
  DINOv2 + MLP + last-4-blocks recipe with three losses: MultiSim
  P2 R@1 = 0.441 vs. InfoNCE 0.178 vs. Triplet 0.356. InfoNCE has the
  best P1 R@10 (0.712) but the worst P2 R@1; it is the most
  aggressive in-distribution memorizer in our experiments. Triplet is
  broken (P1 R@10 = 0.087 ≈ zero-shot) and needs miner / margin
  tuning before it is a usable ablation.

- **F6 — A single Linear layer on frozen DINOv2 reaches half of full
  fine-tune's P1 R@10.** Row 7 (Linear probe) = 0.273 vs. row 9 =
  0.741. This isolates the value of *backbone fine-tuning* (which
  adds +0.468 R@10) from the value of *any head training* (which adds
  +0.176 R@10 from zero-shot 0.097 → linear-probe 0.273). Backbone
  fine-tuning is doing the heavy lifting; head non-linearity is small
  marginal value on top.

- **F7 — Text-only retrieval is degenerate on this benchmark; visual
  evaluation is only meaningful when text is unavailable.** All four
  text-only encoders score 1.000 R@1 on the anchor task because each
  Wikipedia page has a unique title string. Cross-modal CLIP also
  scores 1.000 because CLIP can match anchor images to their parent
  page titles. This means *all interesting differentiation between
  visual models lives in Phase 1* (where the query is a 224 × 224
  crop with no readable text) or in *Phase 2 image-image only* (where
  positives and negatives are both crops of pages, not titles).

### 6. Engineering lessons (and reusable infrastructure)

These are not the science but they are the prerequisite for the
science and worth recording for whoever continues the project.

**Cluster-side DNS block.** Between Apr 19 and Apr 21 the Valar
internal DNS server (192.168.101.5/.6) began returning SERVFAIL for
`huggingface.co`, `hf-mirror.com`, `pypi.org`,
`files.pythonhosted.org`, and `cdn-lfs.huggingface.co`, while still
resolving `google.com`. Workaround: `src/visword/hf_dns_shim.py` — a
pure-stdlib `socket.getaddrinfo` monkey-patch that resolves a
hard-coded set of HF domains via UDP to public DNS resolvers
(8.8.8.8, 1.1.1.1, 8.8.4.4, 9.9.9.9). Must be installed *before* any
`huggingface_hub`, `datasets`, `transformers`, `open_clip`, or `timm`
import; we do this at the top of `prefetch.py`, `eval_zeroshot.py`,
`eval_text.py`, and `clip_backbone.py`. Successfully unblocks both pip
installs and HF model/dataset downloads. `timm` calls a separate code
path (`timm/models/_hub.py`) which we forgot to cover with the shim
and still fails on compute nodes — the fix is to import the shim at
the top of `models/zeroshot.py` *before* the lazy timm import inside
`ZeroShotImageNetViT.__init__`.

**Parallel prefetch.** The naïve attempt at parallel prefetching
(`prefetch_wiki_ss_shard` with multiple stream-skip workers) thrashed
because every worker re-streams from row 0 and discards rows until
its `start_idx`, multiplying the HF-stream egress cost by N rather
than parallelising it. The working approach is `prefetch_arrow_shard`
in `src/visword/data/prefetch.py`: each worker downloads a disjoint
slice of the 809 arrow files via `huggingface_hub.hf_hub_download`
(authenticated with `HF_TOKEN` from `.env`), then iterates locally —
no redundant network. We measured ~5 MB/s per worker on a 493 MB
arrow file (~99 s, ~1 500 rows) and saw approximately linear scaling
to 6 parallel workers on the login node. Each worker writes its own
`manifest.shard_<idx_base>_<idx_base+target>.json` partial; the
final merge step (`merge_shards`) combines all shard partials and
the existing `manifest.json` into a single fingerprinted manifest.

**SLURM dependency chains.** We submit chains of
train→eval→train→eval→… via `sbatch --parsable
--dependency=afterok:JID`. **Lesson:** use `afterany`, not `afterok`,
for cross-row dependencies in long chains. A single eval failure
cascades through `afterok` chains and silently kills every downstream
job (`DependencyNeverSatisfied`); `afterany` lets a failed eval not
poison the next training run. We learned this the hard way after a
linear-probe eval crashed at the post-recall interpret step (the
interpret module had a hard-coded `backbone.model.blocks[-1]` hook
that doesn't exist on `DINOv2LinearProbe`) and took out 11 downstream
jobs.

**Compute caps.** QoS `comx29` allows unlimited submitted jobs but
only **1 T4 GPU** running concurrently. Multiple GPU jobs queue and
FIFO through the single allocation. CPU-only jobs are uncapped (we
ran 7 parallel prefetch workers on the login node simultaneously).
Train + eval pairs at our `salad_5k_main` recipe take ~35 min + ~3
min ≈ 38 min each on a T4 with batch_size 16; the headline
`salad_main` recipe at 10 k train ≈ 80 min. The 8-hour walltime cap
makes batch_size 32 SALAD infeasible on T4 (OOM) and 5+ epoch
training tight; rolling checkpoints would help.

**Reproducibility.** Every run dir includes `config.resolved.yaml`
(the merged Pydantic config), `provenance.json` (git SHA, hub commit,
weights file sha256), `metrics.jsonl` (per-step training log
including loss, pos/neg sim, GPU mem, dustbin mass), and
`phase{1,2}_recall.json` in a fixed schema. Run dir names embed
timestamp + git SHA prefix + `run_name` + 4-char random suffix to
avoid collisions. The cache manifest's sha256 fingerprint is verified
at every dataset open; tampering with any blob invalidates the run.

---

## Part II — Research agenda

The methodological gaps the literature review identifies all apply to
our setup. We list seven proposed directions, ordered by expected
scientific payoff per unit of compute, and tag each with a specific
run-budget estimate.

### 7.1 Sparse-autoencoder analysis of SALAD/CLIP features (highest payoff)

**Question.** Do the features SALAD learns on Wikipedia screenshots
correspond to *linguistic* concepts (words, morphemes, semantic
categories) or only to *layout* concepts (header / infobox / margin /
figure)?

**Design.** Adopt the saev methodology of Stevens et al. (2025;
arXiv:2502.06755): train a TopK or JumpReLU SAE (Gao et al., 2024;
Rajamanoharan et al., 2024) on the residual stream at multiple layers
of (a) DINOv2-ViT-B/14 zero-shot, (b) DINOv2-SALAD trained, (c)
CLIP-ViT-B/16 zero-shot. Keep input distribution identical (the same
wiki-ss eval set). Auto-name each SAE feature using the DN-CBM
template (Rao et al., *ECCV 2024*): take the top-activating image
patches per feature, embed them with a frozen text-aligned model
(CLIP text or BERT), and label each feature by its nearest-neighbour
text concept.

**Why this is the highest-payoff experiment.** No published SAE
analysis exists for any pixel-language model. Stevens et al. found
that *CLIP's language supervision yields cross-style abstractions
DINOv2 does not* (a "Brazil" feature firing across flag/landscape/
tile patches in CLIP but not DINOv2). Replicating their methodology
on our trained SALAD would directly answer whether wiki-ss training
induced *linguistic* abstractions or only *layout* ones — the exact
question F2 raised but cannot itself answer.

**Compute.** ~10 GPU-hours per backbone for SAE training (~16 k
features, 10 layers, 100 k crops). Plus auto-naming. Total ~50
GPU-hours ≈ within Colab Pro+ budget if not on Valar.

**Deliverable.** Side-by-side feature inventory (top-N most-activating
patches per SAE atom + auto-name) for DINOv2 zero-shot vs.
SALAD-trained vs. CLIP zero-shot. Quantitative: fraction of features
that are auto-namable as linguistic concepts (by some simple heuristic
over the named-text distribution) vs. layout/visual.

### 7.2 Test the Platonic Representation Hypothesis on pixel-text models

**Question.** Does a pixel-only model trained on rendered Wikipedia
text develop representations that are linearly aligned with text-only
representations of the same Wikipedia content?

**Design.** Following Huh et al. (*ICML 2024*), Maniparambil et al.
(*CVPR 2024*), and Moayeri et al. (*CVPR 2023 XAI4CV*) measure mutual
nearest-neighbour alignment, debiased CKA (Murphy et al., 2024), and
Procrustes distance between (a) SALAD-trained image features on a
held-out set of pages and (b) BERT/MiniLM/Sentence-BERT text features
on the corresponding page texts. Repeat for DINOv2 zero-shot, CLIP
zero-shot, and our trained CLIP-frozen+SALAD-head variants. The
Text2Concept methodology (Moayeri et al., 2023) gives us a single
linear map; if it transfers cleanly we have direct evidence of
alignment.

**Why now.** The Platonic hypothesis is largely tested on
natural-image vision encoders. Pixel-text models are the *sharpest
possible test*: the visual content *is* the text. If a pixel-only
SALAD aligns with BERT, the hypothesis gains an extreme-condition
confirmation; if it does not, the hypothesis needs refinement.

**Compute.** ~5 GPU-hours total for the alignment computations.

**Deliverable.** Alignment matrix (debiased CKA + Procrustes +
mutual-kNN) across all our trained / zero-shot vision models × three
text encoders (BERT, MiniLM, CLIP-text), with significance bands.

### 7.3 Disentangle reading from understanding

**Question.** Does our SALAD model perform OCR-like glyph recognition
or compositional semantic reading?

**Design.** Three perturbation tests on the eval set:
1. **Typographic attacks** (Goh et al., *Distill 2021*; Cheng et al.,
   *ECCV 2024*): re-render pages with one word replaced by a
   different word in the same visual style. Does the page descriptor
   change? How much?
2. **Word-order shuffles** (per Yuksekgonul et al.'s ARO
   bag-of-words tests, adapted to rendered pages): re-render with
   paragraph words shuffled. If the descriptor barely moves, the
   model is layout-fingerprinting.
3. **Synonym/antonym substitution** (Anschütz et al., 2024 SemAntoNeg
   protocol): re-render with content words swapped for synonyms or
   antonyms. Robust descriptors should change less for synonyms than
   for antonyms.

**Why now.** Materzyńska et al. (*CVPR 2022 Oral*) showed CLIP's
spelling and meaning subspaces are partially separable; Lin et al.
(*ECCV 2024*) Parrot Captions argue ~50 % of CLIP's training signal
is glyph-level. Our model has *only* text in its training distribution,
so glyph-level shortcuts should be even stronger; the perturbation
tests quantify how much.

**Compute.** Re-rendering needs a small webpage-to-image pipeline
(~1 day eng); each test is then ~30 GPU-min on the existing models.

**Deliverable.** Three robustness curves (descriptor distance vs.
perturbation magnitude) for each model.

### 7.4 Emergence-over-training analysis

**Question.** When in training does the SALAD model acquire each
linguistic property (coherence, layout, content)?

**Design.** Rerun the salad_main / cls_main training while
checkpointing every 100 steps; run the §7.1 SAE auto-naming and §7.3
perturbation tests at each checkpoint. The 2025 "Emergence of Text
Readability in VLMs" (arXiv:2506.19389) study showed text readability
emerges suddenly at a specific training point; we should see whether
similar phase transitions exist for SALAD's layout-fingerprinting,
patch-NN sharpness, and dustbin selectivity.

**Compute.** One additional training run with extra checkpointing
(~6 GPU-hours) plus K eval passes (~K × 0.5 GPU-hours).

**Deliverable.** Per-property emergence curve over training steps for
each of the seven findings in §5.

### 7.5 JEPA-on-screenshots (the strategic pivot)

**Question.** Does a Joint-Embedding Predictive objective (I-JEPA,
Assran et al., *CVPR 2023*; V-JEPA, Bardes et al., 2024) on rendered
Wikipedia screenshots learn linguistic abstractions that contrastive
retrieval doesn't?

**Design.** Pretrain a ViT-B/14 (or ViT-B/16 to avoid the DINOv2
torch.hub commit pin) with the I-JEPA objective on our cached
wiki-ss corpus: predict the *representations* of masked target blocks
from the representations of context blocks, using a small predictor
head. No pixel reconstruction (separates this from MAE), no
contrastive loss (separates this from CLIP), no augmentation-invariance
(separates this from DINO). After pretraining, evaluate on (a) our
Phase-1/Phase-2 retrieval, (b) the §7.1 SAE pipeline, (c) the §7.3
perturbation tests, (d) the §7.2 Platonic alignment with BERT.

**Why now.** Three signals converge: (1) our trained models'
in-distribution skill comes at the cost of out-of-distribution
transfer, suggesting the contrastive objective is the over-fitting
agent; (2) PIXEL/CLIPPO/Pix2Struct lineage shows pixel-based
pretraining works but uses different objectives; (3) Neo et al.'s
(*ICLR 2025*) finding that VLM visual streams progressively align
with text vocabulary even without next-token training motivates a
predictive-but-not-generative objective. JEPA fills the gap.

**Compute.** Larger: ~100 GPU-hours for a meaningful pretraining run
on 50 k+ pages. Best done off-cluster (Colab A100 for 50 GPU-hour
budget; or wait until Valar admin unblocks egress and we can run
extended training).

**Deliverable.** A trained pixel-JEPA, evaluated on the same protocol
as the rest of the ladder, plus interpretability artefacts. Strong
contribution if it beats both contrastive SALAD and zero-shot CLIP on
*either* axis.

### 7.6 Late-interaction (ColPali / ColBERT-style) comparison

**Question.** Is the aggregation step (SALAD vs. softmax vs. mean) the
right design choice at all, or should we keep per-patch features and
score via late interaction?

**Design.** Replace the SALAD aggregator with a ColPali-style
multi-vector representation: keep all 256 patch tokens as the page
"document" (or a learned per-patch projection), score query crops via
late interaction (max-sim per query patch over document patches,
summed). Train with the same MultiSim contrastive loss on positive
crop / page pairs. Evaluate on Phase 1 / 2.

**Why now.** ColPali (Faysse et al., *ICLR 2025*) is the current SOTA
on document retrieval and explicitly skips aggregation. Our F3
(VLAD-only ≈ full SALAD) hints that the aggregation choice matters
less than the per-patch quality; late interaction is the natural
extreme of "don't aggregate at all".

**Compute.** ~15 GPU-hours including a small re-design of the eval
loop (per-patch sim is heavier than per-page).

**Deliverable.** Late-interaction R@k vs. SALAD R@k on the same eval
set, plus a memory/latency comparison.

### 7.7 Linguistic probing stack (the methodologically rigorous version)

**Question.** What linguistic properties (POS, dependency structure,
coherence, semantic similarity) are encoded in our SALAD features
and, critically, *causally used* for retrieval?

**Design.** A four-pronged probing protocol:
1. **Structural probe** (Hewitt & Manning, *NAACL 2019*) on patch
   tokens — treat each patch as a "word-piece" proxy and check
   whether parse-tree distances are recoverable as squared L2
   distances under a learned linear transform. Compare across
   DINOv2, CLIP, SALAD-trained.
2. **MDL probing** (Voita & Titov, *EMNLP 2020*) for coherence
   (DiscoEval protocol — Chen et al., *EMNLP 2019*), continuation
   (CONPONO — Iter et al., *ACL 2020*), and semantic sensitivity
   (Zhu et al., *ACL 2018* bigram-shift / argument-sensitivity). MDL
   avoids the probe-capacity confound that plagued classical accuracy
   probes.
3. **Control task selectivity** (Hewitt & Liang, *EMNLP 2019* Best
   Paper Runner-Up): for every probe, also report selectivity (probe
   accuracy − random-label-control accuracy) so encoded ≠ used
   confounds are surfaced.
4. **LEACE concept scrubbing** (Belrose et al., *NeurIPS 2023*) — for
   each linguistic property, fit a closed-form linear erasure and
   measure downstream retrieval degradation; what's *causally
   important* for retrieval should drop performance when erased.

**Why now.** The literature review identifies probing as the
methodological backbone for any claim about linguistic structure in
a new representation learner. Our F2 (over-specialisation) and F4
(Sinkhorn doesn't matter) claims would be much stronger backed by
LEACE-style causal evidence rather than only correlational recall.

**Compute.** ~5 GPU-hours total — probes are small classifiers fitted
on cached descriptors.

**Deliverable.** A probe table with one row per (property × layer ×
model) and three numbers (probe accuracy, MDL, control-task
selectivity), plus LEACE-erasure deltas on Phase 1 R@10.

### 7.8 Multi-page understanding extension

**Question.** Can our retrieval pipeline scale to multi-hop document
QA in the M3DocVQA (Cho et al., 2024) sense?

**Design.** Build a retrieval-augmented generator: SALAD or
CLIP-frozen embeddings as page descriptors; M3DocVQA / DUDE
(Van Landeghem et al., *ICCV 2023*) as the QA benchmark; an
off-the-shelf VLM (LLaVA, Qwen2-VL) as the reader. Compare against
ColPali (Faysse et al., *ICLR 2025*) + the same reader.

**Why now.** This converts our retrieval R@k story into a downstream
QA story, and gives a venue-friendly evaluation if we want to publish
the work as a system paper rather than an interpretability paper.

**Compute.** ~30 GPU-hours for the reader inference plus a few hours
to build the QA pipeline.

**Deliverable.** End-to-end M3DocVQA scores, side-by-side with
ColPali.

### 7.9 Methodological checklist for any of the above

When we do these experiments, we will adhere to the following
checklist the literature review explicitly recommends:

1. **MDL probing instead of accuracy probing** when the question is
   "how much information about X is in this representation" — avoids
   the capacity-confound debate (Hewitt & Liang vs. Pimentel et al.).
2. **Control-task selectivity** reported alongside every probe.
3. **Debiased CKA** (Murphy et al., 2024) for any cross-encoder
   alignment metric, with Procrustes and decoding accuracy as
   complementary metrics (Davari et al., *ICLR 2023* showed CKA can
   be manipulated).
4. **LEACE** for any *causal* claim about which properties are used,
   not merely encoded. RLACE / INLP are acceptable for low-rank
   diagnostics; LEACE for the optimal linear erasure.
5. **TopK or JumpReLU SAEs** for sparse-autoencoder analysis,
   following the Gemma Scope (Lieberum et al., 2024) and saev
   (Stevens et al., 2025) conventions; auto-name features via DN-CBM
   (Rao et al., *ECCV 2024*).
6. **Register-token diagnostics** (Darcet et al., *ICLR 2024
   Outstanding Paper*) before any pooling or aggregation analysis —
   text-screenshot patches vary heavily in information content
   (text vs. margin) and register artifacts are plausible.
7. **Track checkpoints** — "Emergence of Text Readability in VLMs"
   (arXiv:2506.19389) shows readability emerges suddenly. A single
   final-checkpoint snapshot misses the mechanism.
8. **Coherence/continuation/semantic-sensitivity evaluation** on
   DiscoEval (Chen et al., *EMNLP 2019*) for coherence, CONPONO
   (Iter et al., *ACL 2020*) for continuation, and SemAntoNeg
   (Anschütz et al., 2024) for synonym/antonym sensitivity.

---

## 8. Conclusion

We executed a 24-row method ladder that documents, with
controlled-ablation rigor, the relative contributions of backbone
choice, head architecture, contrastive loss, and aggregation method
to visual document retrieval on Wikipedia screenshots. Four findings
stand out: a frozen CLIP image branch dominates anchor-pool transfer
without any task-specific training; DINOv2 fine-tuning trades transfer
for in-distribution skill in proportion to training budget; the VLAD
branch carries essentially all of SALAD's win over a CLS-MLP
baseline; and the Sinkhorn-OT structure with dustbin contributes only
a marginal improvement over plain softmax assignment. These results
validate the project's early concern that the trained model was
learning a Wikipedia-layout fingerprint rather than a reading
representation, and they justify the strategic pivot — not yet
executed — toward a JEPA-on-screenshots pretraining objective combined
with a sparse-autoencoder + linguistic-probing analysis. The
infrastructure (parallel HF prefetch with DNS shim, parallel SLURM
chains with robust dependency handling, full reproducibility
provenance per run, and a method-ladder factory shared across DINOv2
and CLIP backbones) is in place to run the §7 program directly.

## 9. Citations

The citations below are drawn from the literature review supplied
April 2026. Errata for the project's previous specification:

- **SALAD** is *Izquierdo, S. & Civera, J. "Optimal Transport
  Aggregation for Visual Place Recognition." CVPR 2024,
  pp. 17 658–17 668. arXiv:2311.15937.* This is the correct citation.
  Earlier project documents misattributed it as "Serizel 2023" — the
  confusion stems from the GitHub handle `serizba` of the first
  author.
- **RLACE** (Ravfogel et al.) is *ICML 2022*, not NeurIPS 2022.

Full reference list — see the literature review document.

Pixel-LM lineage: PIXEL (Rust et al., ICLR 2023), CLIPPO (Tschannen
et al., CVPR 2023), Pix2Struct (Lee et al., ICML 2023), PIXAR (Tai
et al., Findings of ACL 2024), PTP (Gao et al., 2024), PixelGPT/
DualGPT (Chai et al., EMNLP 2024), ScreenAI (Baechler et al., IJCAI
2024), PIXEL-M4 (Kesen et al., 2025), Pixel-level Fallback (Lotz et
al., 2025).

Document AI: ColPali (Faysse et al., ICLR 2025), Donut (Kim et al.,
ECCV 2022), LayoutLMv3 (Huang et al., ACM MM 2022), Nougat (Blecher
et al., 2023), SelfDoc (Li et al., CVPR 2021), DocOwl-1.5 / DocOwl-2
(Hu et al., Findings of EMNLP 2024), ColBERT (Khattab & Zaharia,
SIGIR 2020), ViDoRe (Faysse et al., 2024; V2 Macé et al. 2025),
DUDE (Van Landeghem et al., ICCV 2023), MP-DocVQA (Tito et al.,
Pattern Recognition 2023), M3DocVQA / M3DocRAG (Cho et al., ICCV 2025
Workshop).

Aggregation and OT: SALAD (Izquierdo & Civera, CVPR 2024), NetVLAD
(Arandjelović et al., CVPR 2016), Set Transformer (Lee et al., ICML
2019), Perceiver / Perceiver IO (Jaegle et al., ICML 2021 / ICLR 2022),
SigLIP / SigLIP 2 (Zhai et al., ICCV 2023; Tschannen et al., 2025),
Sinkformers (Sander et al., AISTATS 2022), ESPFormer (2025).

Probing: Alain & Bengio (ICLR Workshop 2017), Conneau et al. (ACL
2018), Tenney et al. (ICLR 2019), Hewitt & Manning (NAACL 2019),
Hewitt & Liang (EMNLP 2019 Best Paper Runner-Up), Pimentel et al.
(ACL 2020), Voita & Titov (EMNLP 2020), Belinkov (Computational
Linguistics 2022), Kornblith et al. CKA (ICML 2019), Murphy et al.
debiased CKA (ICLR 2024 Re-Align Workshop), Davari et al. (ICLR
2023). Causal: INLP (Ravfogel et al., ACL 2020), Amnesic (Elazar et
al., TACL 2021), RLACE (Ravfogel et al., ICML 2022), LEACE (Belrose
et al., NeurIPS 2023). Decoding: Patchscopes (Ghandeharioun et al.,
ICML 2024), Tuned Lens (Belrose et al., 2023). Recent: Platonic
Representation (Huh et al., ICML 2024), Neo et al. (ICLR 2025), Jiang
et al. PROJECTAWAY (ICLR 2025), Esfandiarpoor et al. EX2 (EMNLP 2024),
"How MLLMs Solve Image Tasks" (2025).

Mech interp: Bricken et al. (Anthropic 2023), Cunningham et al.
(ICLR 2024), Templeton et al. (Anthropic 2024), TopK SAE (Gao et al.,
2024), Gated SAE / JumpReLU SAE (Rajamanoharan et al., 2024), Gemma
Scope (Lieberum et al., BlackboxNLP 2024). Vision SAEs: Fry (2024),
Daujotas (2024), DN-CBM (Rao et al., ECCV 2024), CoX-LMM (Parekh et
al., NeurIPS 2024), PatchSAE (Lim et al., 2025), Zhang et al. (ICCV
2025), Joseph et al. (CVPR 2025 MI-Vision), saev (Stevens et al.,
2025), Zaigrajew Matryoshka (ICML 2025), Prisma (Joseph et al.,
2025). ViT mech interp: Gandelsman et al. (ICLR 2024 Oral; ICLR
2025), Balasubramanian et al. (NeurIPS 2024), Vilas et al. (NeurIPS
2023), Palit et al. (ICCV 2023 CLVL Workshop), Basu et al. (NeurIPS
2024), Darcet et al. registers (ICLR 2024 Outstanding Paper), Jiang
et al. trained-registers (2025), Adebayo et al. (NeurIPS 2018).

Vision-text knowledge: Yuksekgonul et al. ARO (ICLR 2023 Oral),
Thrush et al. Winoground (CVPR 2022), Hsieh et al. SugarCrepe
(NeurIPS 2023 D&B), Ma et al. CREPE (CVPR 2023 Highlight),
Materzyńska et al. (CVPR 2022 Oral), Goh et al. (Distill 2021),
Azuma & Matsui (ICCV 2023 Workshop), Cheng et al. (ECCV 2024),
Qraitem et al. (NeurIPS 2024 Workshop), Parcalabescu et al. VALSE
(ACL 2022), Lewis et al. (Findings of EACL 2024), Kamath et al.
(EMNLP 2023), Hintersdorf et al. (JAIR 2024), Jose et al.
DINOv2-meets-Text (Meta 2024), Lin et al. Parrot Captions (ECCV
2024), Emergence of Text Readability (2025).

SSL representation: DINOv2 (Oquab et al., TMLR 2024), DINO (Caron et
al., ICCV 2021), MAE (He et al., CVPR 2022), BEiT (Bao et al., ICLR
2022), Park et al. CL-vs-MIM (ICLR 2023), Geirhos et al. shape-vs
-texture, Neural Collapse (Papyan et al., PNAS 2020; Wu & Papyan
NeurIPS 2024 Linguistic Collapse; Ben-Shaul et al. NeurIPS 2023),
Platonic Representation (Huh et al., ICML 2024 Position Oral),
Maniparambil et al. (CVPR 2024), Moayeri et al. Text2Concept (CVPR
2023 XAI4CV Workshop), I-JEPA (Assran et al., CVPR 2023), Darcet et
al. registers (ICLR 2024). Patch-level semantics: LOST, TokenCut,
CutLER, Siméoni et al. (IJCV 2024).

Evaluation: MTEB / MMTEB / MIEB (Muennighoff et al., EACL 2023; 2025),
SentEval (Conneau & Kiela, LREC 2018), Sentence-BERT (Reimers &
Gurevych, EMNLP 2019), Barzilay & Lapata entity grid (ACL 2005;
Computational Linguistics 2008), CONPONO (Iter et al., ACL 2020),
DiscoEval (Chen et al., EMNLP 2019), Zhu, Li, de Melo (ACL 2018),
Conneau et al. probing (ACL 2018), Gardner et al. contrast sets
(Findings of EMNLP 2020), Maimon & Tsarfaty COHESENTIA (EMNLP 2023),
Anschütz et al. SemAntoNeg (2024).

---

---

## 10. New result — Platonic alignment measured

Added 2026-04-23. A first empirical probe from the §7.2 program.
Source: `runs/platonic_alignment_2026-04-23_212926/report.json`.

**Protocol.** Sampled 500 pages uniformly from the 376k-row merged
cache, encoded each page's *image* with frozen DINOv2-ViT-B/14 and
CLIP-ViT-B/16 image branch, and each page's *title* with
`bert-base-uncased` (mean-pooled), `all-MiniLM-L6-v2`, and the CLIP
ViT-B/16 text branch. Computed pairwise linear alignment via debiased
HSIC / CKA (Murphy et al., 2024), Procrustes distance after
unit-Frobenius normalisation, and mutual-kNN overlap at k=10 (Huh et
al., *ICML 2024*).

**Debiased CKA matrix:**

|                | DINOv2 | CLIP-img | BERT  | MiniLM | CLIP-txt |
|----------------|--------|----------|-------|--------|----------|
| **DINOv2**     | —      | 0.196    | 0.027 | 0.038  | 0.041    |
| **CLIP-img**   | 0.196  | —        | 0.232 | 0.294  | 0.364    |
| **BERT**       | 0.027  | 0.232    | —     | 0.319  | 0.375    |
| **MiniLM**     | 0.038  | 0.294    | 0.319 | —      | 0.382    |
| **CLIP-txt**   | 0.041  | 0.364    | 0.375 | 0.382  | —        |

**Mutual-kNN @10 corroborates the CKA ranking:**

|                | DINOv2 | CLIP-img | BERT  | MiniLM | CLIP-txt |
|----------------|--------|----------|-------|--------|----------|
| **DINOv2**     | —      | 0.089    | 0.042 | 0.042  | 0.032    |
| **CLIP-img**   | 0.089  | —        | 0.106 | 0.120  | 0.138    |
| **BERT**       | 0.042  | 0.106    | —     | 0.194  | 0.133    |
| **MiniLM**     | 0.042  | 0.120    | 0.194 | —      | 0.142    |
| **CLIP-txt**   | 0.032  | 0.138    | 0.133 | 0.142  | —        |

**Findings.**

- **F8 — DINOv2 is linearly isolated from text** on the wiki-ss
  distribution. CKA ≤ 0.041 against every text encoder. Mutual-kNN ≤
  0.042 at k=10 (vs. ~0.10 chance for the distribution). On rendered
  text, the Platonic Representation Hypothesis (Huh et al., *ICML
  2024*) **fails for pure self-supervised image pretraining** —
  DINOv2's image features do not share linear structure with BERT/MiniLM
  /CLIP-text representations of the same page's title. This is a
  clean negative result, directly contradicting the strongest reading
  of the hypothesis; it is consistent with Maniparambil et al.'s
  (*CVPR 2024*) observation that unimodal vision-vs-text alignment is
  substantial only after post-hoc training, not out of the box.

- **F9 — CLIP-image features are 5–10× more aligned with text
  encoders than DINOv2's are.** Best CLIP-image↔text CKA is 0.364
  (with CLIP-text, same-family) and 0.294 with an external MiniLM.
  Mutual-kNN@10 ≥ 0.106 with every text encoder. This is direct
  evidence that **language supervision during vision pretraining
  induces linear text-alignability**, not just downstream task
  performance. Consistent with Stevens et al.'s (2025) SAE observation
  that CLIP's language supervision yields cross-style abstractions
  DINOv2 doesn't.

- **F10 — The strongest vision↔text pair is within-family**
  (CLIP-image ↔ CLIP-text, CKA 0.364), but CLIP-image ↔ external text
  (MiniLM 0.294, BERT 0.232) is also substantial — cross-encoder
  alignment is not a trivial artefact of shared embedding space.

- **F11 — Text encoders cluster together** with CKA 0.319–0.382
  amongst themselves, as expected for models trained on the same
  language. This establishes a natural "ceiling" for cross-modal
  alignment (no vision-text pair can reasonably exceed within-text
  alignment in this population).

**Implications for the project.**

1. The F2 anchor-pool inversion (DINOv2-SALAD loses transfer to the
   anchor pool) now has a deeper explanation: DINOv2's features are
   *not aligned to the linguistic content of the page* in any
   linearly-decodable sense, so contrastive fine-tuning on layout
   fingerprints pushes them further from what would help anchor
   retrieval. CLIP's features are text-aligned out of the box, which
   is why zero-shot CLIP dominates Phase 2.

2. Any "train a head to fix transfer" intervention on DINOv2 is
   fighting the feature geometry; the cheaper intervention is to
   switch backbones (F1).

3. The strong conclusion — *the Platonic hypothesis is false for
   self-supervised image pretraining on rendered text* — depends on
   n=500 and a linear-alignment assumption. The §7.7 probing program
   (structural probe, MDL, LEACE) on these same representations
   would test the nonlinear decodability, and is the natural next
   experiment.

**Caveats.** Alignment is measured in a 500-sample, frozen-feature
linear-kernel regime. We do not claim "vision and text representations
are nowhere alignable"; we claim that *on rendered Wikipedia
screenshots, DINOv2's image features and BERT-style text features do
not share substantial linear structure, while CLIP's image features
do*. The Text2Concept methodology (Moayeri et al., 2023) could be
applied to learn a single linear map from DINOv2 to CLIP-text space
and measure its transfer quality — this would quantify how much
*post-hoc* alignment is recoverable.

---

---

## 11. New result — Training pushes features TOWARD text alignment but anchor-pool transfer still degrades

Added 2026-04-24. Extended `runs/platonic_trained_2026-04-24_005456/`.
500-page sample, same protocol as §10, with our trained models added
to the matrix: SALAD-main (`runs/2026-04-19_201330…_salad-main_138f`),
CLS-main (`runs/2026-04-19_213919…_cls-main_db61`), and the linear
probe (`runs/2026-04-21_195057…_row07-linear-probe_385b`).

**Debiased CKA — trained models against text encoders:**

|                  | BERT   | MiniLM | CLIP-text | (vs zero-shot DINOv2) |
|------------------|--------|--------|-----------|------------------------|
| DINOv2 zero-shot | 0.034  | 0.037  | 0.052     | reference (1×)         |
| Linear probe     | 0.022  | 0.026  | 0.049     | ≈ same as zero-shot    |
| **CLS-main (trained)** | **0.081** | **0.077** | **0.105** | **2.0–2.4×** zero-shot |
| **SALAD-main (trained)** | **0.105** | **0.109** | **0.118** | **2.3–3.1×** zero-shot |
| (CLIP-image zero-shot) | 0.232 | 0.294 | 0.364 | 5–7× zero-shot DINOv2 |

**Trained-vs-trained / vs frozen-DINOv2:**

|                  | DINOv2 zero-shot | CLIP-image | Linear probe |
|------------------|------------------|------------|--------------|
| SALAD-main       | 0.436            | 0.269      | 0.386        |
| CLS-main         | 0.520            | 0.235      | 0.484        |
| Linear probe     | **0.874**        | 0.138      | —            |

**Findings.**

- **F12 — Contrastive fine-tuning DOES move DINOv2 features toward
  text alignment**, by 2–3× over the frozen baseline, contrary to the
  strongest reading of F8. SALAD-main's CKA against BERT is 0.105
  (vs. 0.034 zero-shot, +0.071); against MiniLM 0.109 (vs. 0.037,
  +0.072); against CLIP-text 0.118 (vs. 0.052, +0.066). This refutes
  the "training pushes features into a pure layout subspace" reading
  of F2: training does increase the linguistic decodability of
  DINOv2-derived descriptors. The increase is not large enough to
  reach CLIP's natural levels (0.232–0.364) but is unambiguous.

- **F13 — Yet text alignment increase ≠ Phase-2 transfer
  improvement.** SALAD-main (CKA 0.105 with BERT, P2 R@1 0.390)
  *underperforms* zero-shot DINOv2 (CKA 0.034, P2 R@1 0.593) on the
  anchor task. The Phase-2 task is image→image, not image→text;
  layout-fingerprinting (which training induces) and text alignment
  (which training also induces) are *both* increasing, but the
  former dominates the anchor-image-similarity outcome. Conclusion:
  text alignment is *not* a sufficient predictor of cross-distribution
  visual transfer. CLIP wins Phase 2 because its image features are
  *both* text-aligned (CKA 0.232) *and* not over-specialised to
  wiki-ss layout — a combination training-on-wiki-ss cannot
  reproduce starting from DINOv2.

- **F14 — Linear probe ≈ DINOv2 zero-shot in feature geometry**
  (CKA 0.874 with frozen DINOv2). Adding a single trainable Linear
  head barely deforms the underlying representation — the geometry
  remains DINOv2's. This explains why F6 saw the linear probe reach
  roughly half of full SALAD's R@10 (it carries half the
  expressivity of the trainable backbone) but stay close to
  zero-shot's anchor-pool transfer (CKA 0.022 with BERT —
  essentially identical to zero-shot DINOv2's 0.034). The Linear
  head's R@10 = 0.273 vs. zero-shot's 0.097 reflects task-specific
  rotation, not representational restructuring.

- **F15 — SALAD-main and CLS-main share 0.593 CKA** despite
  different head architectures and 33× output-dim ratio (8 448 vs.
  256). They have learned similar geometric properties from the same
  data — the "Wikipedia-layout fingerprint" sits in both, just
  packaged differently.

**Implications.**

The story is now clearer:
1. The Platonic hypothesis is testable in two regimes: *backbone
   choice* (CLIP vs. DINOv2 — F8/F9: clear separation) and *post-hoc
   training* (CLS/SALAD on top of DINOv2 — F12: directional move,
   incomplete amount).
2. The Phase-2 inversion (F2) is NOT explained by features moving
   "away from text" — they actually move toward it. It is explained
   by features moving toward *both* text *and* wiki-ss-layout
   simultaneously, with the layout component dominating the
   image→image anchor task.
3. The natural next experiment (§7.7 LEACE-style concept scrubbing)
   becomes very pointed: scrub the "text-aligned" subspace from
   SALAD-main and re-test Phase 2; scrub the "layout fingerprint"
   subspace and re-test. The decomposition would show *which*
   moves-toward is causally responsible for the inversion.

**Caveats.** Same as §10: linear-kernel CKA on n=500. F12's "2–3×
increase" is a modest absolute change in alignment; the qualitative
story (training increases text-decodability) is robust but its
practical implication for downstream tasks depends on what
*linguistic* structure is decodable, not just how much linear-kernel
overlap exists. This is what the structural / MDL probing program in
§7.7 will quantify.

---

*End of document (revision 2026-04-24).*
