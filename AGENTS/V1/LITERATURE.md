# V1 — Literature notes: high-frequency text & global understanding in vision

Grounding for the reading/resolution design (E1, E4-MAE, E6 global, E7 high-freq) and a
ready source for the paper's related-work. Claims are cited; verify before quoting in the
paper.

## Why text is hard for vision encoders

1. **Patch embedding is low-pass.** A ViT patch-embed is one linear projection per patch →
   ~16× spatial downsample; high-frequency components (glyph edges) are discarded. Mitigations:
   multi-scale / adaptive patch embedding ([MSPE, NeurIPS'24](https://proceedings.neurips.cc/paper_files/paper/2024/file/3396657fe1a3c9a43ac7cd809c51a41e-Paper-Conference.pdf);
   [Adaptive Patch Sizes](https://arxiv.org/html/2510.18091v1)). Self-attention is itself a
   low-pass operation, so depth does not restore the highs.
2. **Resolution vs token budget.** Reading body text needs many pixels *and* many tokens per
   glyph → quadratic cost. This is the wall every document model hits.
3. **Tokenizer/VAE bottleneck** — if you compress through a latent first, small text dies there.

## The VAE / autoencoder thread (informs E7)

- **Latent VAEs destroy small text:** "When channel capacity is limited, fine details (small
  text, textures) are lost at the tokenizer stage — no amount of [downstream] training can
  recover them" ([DA-VAE](https://caixin98.github.io/davae/)). This is why latent-diffusion
  models render garbled text while pixel-space models do better.
- **Reconstruction↔generation trade-off:** more latent channels → better reconstruction (text
  survives) but harder latents ([Frequency Perspective](https://arxiv.org/html/2511.22249v1)).
  Structured fixes: base+detail channels with an alignment loss ([DA-VAE](https://caixin98.github.io/davae/));
  wavelet high-frequency sub-space ([Latent Wavelet Diffusion](https://arxiv.org/pdf/2506.00433)).
- **Discrete tokenizers (VQ-VAE/dVAE/VQGAN)** share the text problem; fixed with localized
  text/face perceptual losses ([InsightTok](https://arxiv.org/html/2605.14333)).
- **Implication:** do NOT put a lossy latent VAE in front of text. Stay pixel/near-pixel or
  frequency-structured; if learning an AE, add a base+detail split or a text-perceptual loss.

## Document-AI: resolution + high-frequency handling (informs E1, E3)

- **Aspect-preserving variable-resolution patching** — [Pix2Struct](https://arxiv.org/html/2210.03347)
  maximizes patches within a budget, never square-resizing ("severe aspect-ratio distortions");
  needs ~1M pixels, diminishing returns after. (Validates E1/D19: the squash, not 224 itself,
  was the killer.)
- **Hierarchical encoders** — Donut/Nougat use Swin to afford high resolution cheaply.
- **Frequency-domain documents** — [DocPedia](https://arxiv.org/pdf/2311.11810) processes pages
  in the **DCT** domain to take very high resolution (~2560²) without a token explosion, keeping
  high-frequency text coefficients. (Top candidate for E7 option 1.)

## Global understanding: fuse global + local, don't pick one (informs E6)

- Image retrieval long combines a **global descriptor with local features**
  ([DALG](https://arxiv.org/pdf/2207.00287); [patch-as-local-features](https://openaccess.thecvf.com/content/ACCV2022/papers/Phan_Patch_Embedding_as_Local_Features_Unifying_Deep_Local_and_Global_ACCV_2022_paper.pdf)).
  Our SALAD aggregator is exactly a global descriptor pooled from local patch tokens.
- **Multi-vector late interaction** keeps all local tokens (ColPali; [layout-informed
  multi-vector](https://arxiv.org/pdf/2603.01666)) — we already have this protocol.
- **Document global-local fusion** — [GlobalDoc](https://arxiv.org/pdf/2309.05756): "synergistic
  global-local fusion is significantly more effective than simple global vector inclusion"; the
  global context *re-weights* the local features. → E6 should condition, not concatenate.

## MAE for text (informs E4, E7)

- MAE's **pixel reconstruction preserves high-frequency detail** better than feature-target
  objectives — the basis of our I-JEPA (feature-prediction) vs MAE (pixel-reconstruction)
  contrast. Caveat: the MAE *encoder* still uses the low-pass linear patch-embed, so pair with a
  conv stem / smaller patches if reading is weak. Frequency/scale-aware variants exist
  ([FreqMAE](https://par.nsf.gov/servlets/purl/10547297);
  [Scale-MAE](https://openaccess.thecvf.com/content/ICCV2023/papers/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.pdf)).

## One-line implications for our tickets
- **E1 (D19):** native crops (no downsample) — confirmed correct by Pix2Struct/Donut.
- **E4 (MAE reader):** pixel-objective is high-freq-friendly; watch the patch-embed low-pass.
- **E6 (global):** conditioning/re-weighting fusion (GlobalDoc), not concat.
- **E7 (high-freq):** DCT-input (DocPedia) or conv/wavelet stem; never a lossy latent VAE.
