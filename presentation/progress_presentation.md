# Slide 1: Can Vision Models Read? (0:00 - 0:30)
## Can Vision Models Read?
### A Study of Visual Document Retrieval and Latent Text Representations
* **Authors:** Barış Cem Bakay, Halil İbrahim Kanpak
* **Affiliation:** Department of Computer Engineering, Koç University
* **Goal:** Present visual page retrieval benchmarks across vision pretrainings, investigate the layout shortcut confound, and scale self-supervised predictive encoders (I-JEPA) to native resolution.

---

# Slide 2: Motivation & Research Questions (0:30 - 1:15)
## OCR-Free Visual Retrieval
* **The Goal:** Bypass traditional text pipelines (OCR + text encoders) to retrieve document pages directly using page screenshots, preserving layout, figures, and formatting.
* **Core Research Inquiries:**
  * **Q1 (Zero-Shot Encoders):** Which vision pretrainings inherently align with visual text features zero-shot?
  * **Q2 (Platonic Hypothesis):** Do vision space and text space align representationally as model scale grows?
  * **Q3 (Unsupervised Reading):** Does image-only predictive pretraining (I-JEPA) encode latent linguistic features?
  * **Q4 (The Shortcut Confound):** Do models actually read body text, or do they exploit layout cues (like titles)?

---

# Slide 3: Phase 1 Recap: Zero-Shot & The Platonic Hypothesis (1:15 - 2:00)
## Initial Findings (Progress Report Status)
* **Zero-Shot Visual Retrieval (Protocol A, 2k pages):**
  * Image-text contrastive models (CLIP, SigLIP) dominate zero-shot page retrieval (CLIP: R@10 = 0.779).
  * Self-supervised and image-only models (DINOv2, I-JEPA) sit near chance levels zero-shot.
* **Testing the Platonic Representation Hypothesis:**
  * Measured mutual-kNN@10 alignment between vision features and BERT text embeddings.
  * Contrastive features show strong text alignment (0.09 - 0.14); image-only features align poorly (0.007 - 0.027).
  * **Spearman's \rho = 0.83** ($p = 0.04$) between text-alignment and retrieval recall, confirming that semantic alignment predicts retrieval success.

---

# Slide 4: Phase 1 Recap: Fine-Tuning & The Title Confound (2:00 - 2:45)
## Shortcut Learning vs. LEACE Erasure
* **The Fine-Tuning Paradox:** Fine-tuning DINOv2-SALAD-50k beats zero-shot CLIP, reaching **R@10 = 0.915**.
* **Title-Blanking Experiment:** Blanking the top 15% (title region) of pages:
  * Fine-Tuned DINOv2: Retrieval recall crashes by **-0.242** (0.915 \rightarrow 0.673).
  * Zero-Shot CLIP: Recall increases by **+0.073** (0.779 \rightarrow 0.852).
  * *Finding:* Fine-tuning teaches DINOv2 to overfit to spatial layout coordinates (titles), whereas CLIP learns translation-invariant text patterns.
* **LEACE Causal Title-Erasure:** 
  * Projecting out the top "title direction" axis in feature space boosts CLIP retrieval to **0.863** (beating blanking pixels!).
  * Erasing it from DINOv2 has minimal effect, proving DINOv2 does not represent titles as a distinct linear feature.

---

# Slide 5: Phase 1 Recap: I-JEPA's Latent Text Alignment (2:45 - 3:30)
## Unlocking Unsupervised Text Representation
* **The Representation Coordinate Problem:**
  * I-JEPA features have zero-shot text-alignment near chance (R@10 = 0.015).
  * However, is semantic text structure present but represented in the wrong basis?
* **Linear Projection Adapter:**
  * We trained a single linear projection (InfoNCE on 4,000 screenshot-title pairs) from I-JEPA to BERT space.
  * Adapted I-JEPA text alignment jumped by **13x** to **R@10 = 0.197** (no encoder fine-tuning, no nonlinearities).
  * *Conclusion:* Image-only predictive pretraining does learn text-aligned structure, but it is latent and requires an alignment adapter to recover.

---

# Slide 6: Main Focus: The Resolution Bottleneck & Cropping (3:30 - 4:15)
## Shifting to Native Legibility (New Progress)
* **The Scale-Down Confound:** Phase 1 evaluations scaled 490x490 page crops down to 224x224.
  * This 2.19x shrink rendered body text sub-pixel (illegible to human and OCR models).
  * *Implication:* Models in Phase 1 were structurally prevented from reading body text, forcing them to rely on title shortcuts.
* **The Solution: Legible-Cropping Protocol:**
  * Implemented a `TextAwareCropper` that processes page tiles at native 490x490 resolution (1,225 tokens per crop).
  * Applies line-snapping heuristics to keep line boundaries intact.
  * Ensures that characters are fully legible for reading.

---

# Slide 7: Main Focus: Re-Baselining the Grid & Bounds (4:15 - 5:00)
## Performance at Native Resolution
* **The Perfect-Text Upper Bound:**
  * Mean-pooling BERT text embeddings directly from the page text yields **R@10 = 0.938**.
  * Represents the text-only retrieval ceiling.
* **Contrastive MAE Reader Baseline:**
  * A strong custom-trained baseline using MAE features to reconstruct/embed text.
  * Reaches **R@10 = 0.098** at native resolution.
* **Re-baselined Vision Encoders:**
  * Evaluated standard backbones at 490x490 resolution.
  * Contrastive vision encoders retain their dominance, establishing a clean baseline for text retrieval.

---

# Slide 8: Main Focus: Visual Document-Pretrained Encoders (5:00 - 5:45)
## Evaluating Document AI Models Zero-Shot
* **The Document-Pretrained Family:**
  * Evaluated visual encoders explicitly pretrained on documents or document OCR tasks.
* **Nougat (Swin Transformer):**
  * Reaches **R@10 = 0.049** zero-shot.
  * Shows that even models trained on document layouts require explicit retrieval adaptation.
* **Pix2Struct (Google):**
  * Custom forward pass modification implemented using `Pix2StructImageProcessor` to bypass the default 224x224 input squashing.
  * Zero-shot evaluation active on GPU node `ai26` to establish the zero-shot ceiling for document-reading models.

---

# Slide 9: Main Focus: Cross-Modal Pretraining - I-JEPA with Text Targets (5:45 - 6:45)
## Predicting Language from Unmasked Image Context
* **Pretraining Objective:** Replace standard visual feature prediction with a cross-modal (text-target) objective during I-JEPA pretraining.
* **How it works:**
  * The context encoder processes unmasked screenshot patches.
  * The predictor (`VisionTransformerTextPredictor`) takes context patch representations and mask query tokens (with 1D positional embeddings for text).
  * It predicts the **BERT semantic embeddings** of the text rendered in the masked visual regions.
* **Pretrained Backbone:** Trained a ViT-H/14 backbone (`ijepa-text-target-all-blocks-ful_e27e`) on 90,000 Wikipedia pages using body-text target tokens at native 490x490 resolution.
* **Significance:** Forces the vision encoder to learn semantic text structures directly from visual context, laying the groundwork for native-resolution document retrieval.

---

# Slide 10: Main Focus: Ongoing Experiment - Native-Resolution Training Grid (6:45 - 7:30)
## Probing High-Resolution Latent Emergence on Cluster
* **The Goal:** Benchmark standard image-only pretraining against our cross-modal text-target pretraining, and evaluate retrieval head configurations.
* **The 4-GPU Cluster Training Grid:**
  * Running 4 parallel training runs on A40 GPUs using native 490x490 resolution (1,225 tokens per crop).
* **Experimental Configuration Matrix:**
  * **Runs 1 & 2 (Image-Only Backbone):** Frozen/unfrozen I-JEPA backbone pretrained on images, comparing MLP vs. SALAD retrieval heads.
  * **Runs 3 & 4 (Text-Target Backbone):** Unfrozen cross-modal I-JEPA backbone (`ijepa-text-target-all-blocks-ful_e27e`), comparing MLP vs. SALAD retrieval heads.
  * *Target:* Validate if text-target pretraining provides a stronger starting point for visual retrieval than pure image pretraining.

---

# Slide 11: Main Focus: Confound Control via Title Masking (7:30 - 8:15)
## Blocking Layout Shortcuts during Training
* **The Goal:** Force the model to read the actual body text instead of exploiting spatial layout shortcuts (the "title layout cheat").
* **Random Title Masking:**
  * Apply a random whiteout mask over the top 15% (title region) of page screenshots during fine-tuning.
  * Eliminates the title layout fingerprint from the training distribution.
* **Initial Verification:**
  * Forces the model to extract and match semantic tokens from the body text.
  * Prevents cross-set transfer performance drops.

---

# Slide 12: Main Focus: Advanced Retrieval Architectures (8:15 - 9:00)
## Multi-Vector & Multi-Scale Vision
* **Two-Stream Global-Local Reader:**
  * A two-stream visual pipeline designed to solve the trade-off between local legibility and global layout context.
  * *Stream 1:* High-resolution native crops (490x490) for reading text.
  * *Stream 2:* Heavily downsampled global overview (224x224) to maintain layout structure.
* **Late Interaction (ColPali):**
  * Evaluating state-of-the-art multi-vector models (ColPali) using MaxSim scoring.
  * Testing if single-vector models (SALAD) can reach multi-vector performance.

---

# Slide 13: Conclusion & Takeaways (9:00 - 10:00)
## Summary of Work
* **Phase 1 Recap:** Pretraining objective determines zero-shot retrieval. Contrastive dominates; image-only has latent text structure but exploits layout coordinates when fine-tuned.
* **Phase 2 Contribution:** Resolution is the reading bottleneck. We introduced the legible 490x490 cropping protocol to enable true reading.
* **Ongoing Work:** Re-baselined standard encoders (Nougat, MAE), and launched 4 parallel GPU runs to probe I-JEPA at native resolution.
* **Future Vision:** Combining title masking (to block layout shortcuts) with two-stream networks and late interaction (ColPali) to build robust OCR-free page search.
