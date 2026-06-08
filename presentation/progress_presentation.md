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
  * **Q5 (Text-Target JEPA):** Can cross-modal (Text-Target) pretraining inside I-JEPA construct stronger text representations than pure image-only pretraining?

[FIGURE: Visual Retrieval Pipeline vs. Standard OCR + Text Retrieval]

---

# Slide 3: Phase 1 Recap: Zero-Shot & The Platonic Hypothesis (1:15 - 2:00)
## Initial Findings (Progress Report Status)
* **Zero-Shot Visual Retrieval (Protocol A, 2k pages):**
  * Image-text contrastive models (CLIP, SigLIP) dominate zero-shot page retrieval.
  * Self-supervised and image-only models (DINOv2, I-JEPA) sit near chance levels zero-shot.
* **Testing the Platonic Representation Hypothesis:**
  * Measured mutual-kNN@10 alignment between vision features and BERT text embeddings.
  * Contrastive features show strong text alignment; image-only features align poorly.
  * Strong correlation between text-alignment and retrieval recall, confirming that semantic alignment predicts retrieval success.

[FIGURE: Scatter plot of vision-text mutual-kNN alignment vs. Zero-shot Recall@10 (Spearman rho = 0.83)]

---

# Slide 4: Phase 1 Recap: Fine-Tuning & The Title Confound (2:00 - 2:45)
## Shortcut Learning vs. LEACE Erasure
* **The Fine-Tuning Paradox:** Fine-tuning DINOv2-SALAD beats zero-shot CLIP, but relies on layout shortcuts.
* **Title-Blanking Experiment:** Blanking the top 15% (title region) of pages:
  * Fine-Tuned DINOv2: Retrieval recall crashes.
  * Zero-Shot CLIP: Recall increases, proving it reads text invariants rather than matching layouts.
* **LEACE Causal Title-Erasure:** 
  * Projecting out the top "title direction" axis in feature space boosts CLIP retrieval, proving titles exist as a structured linear axis in CLIP's representation.
  * Erasing it from DINOv2 has minimal effect, confirming DINOv2 does not represent titles as distinct text concepts.

[FIGURE: Bar chart of Retrieval Recall under Title Blanking and LEACE Causal Erasure (CLIP vs. DINOv2)]

---

# Slide 5: Phase 1 Recap: I-JEPA's Latent Text Alignment (2:45 - 3:30)
## Unlocking Unsupervised Text Representation
* **The Representation Coordinate Problem:**
  * Image-only I-JEPA features have zero-shot text-alignment near chance.
  * However, is semantic text structure present but represented in a different visual basis?
* **Linear Projection Adapter:**
  * Training a simple linear projection adapter maps I-JEPA features into BERT text space.
  * Adapted I-JEPA text alignment improves by 13x, showing that image-only predictive pretraining does learn latent text structures.

[FIGURE: Diagram of the Linear Projection Adapter from I-JEPA visual features to BERT text space]

---

# Slide 6: Main Focus: The Resolution Bottleneck & Cropping (3:30 - 4:15)
## Shifting to Native Legibility (New Progress)
* **The Scale-Down Confound:** Phase 1 evaluations scaled 490x490 page crops down to 224x224.
  * This 2.19x shrink rendered body text sub-pixel and illegible.
  * *Implication:* Models in Phase 1 were structurally prevented from reading body text, forcing them to rely on layout shortcuts.
* **The Solution: Legible-Cropping Protocol:**
  * Process page tiles at native 490x490 resolution (1,225 tokens per crop).
  * Apply line-snapping heuristics to keep line boundaries intact.
  * Ensures that characters are fully legible for reading.

[FIGURE: Visual comparison of 224x224 squashed crops (illegible) vs. 490x490 native crops (legible)]

---

# Slide 7: Main Focus: Re-Baselining the Grid & Bounds (4:15 - 5:00)
## Performance at Native Resolution
* **The Perfect-Text Upper Bound:**
  * Mean-pooling BERT text embeddings directly from the page text.
  * Represents the text-only retrieval ceiling.
* **Contrastive MAE Reader Baseline:**
  * A strong custom-trained baseline using MAE features to reconstruct/embed text.
* **Re-baselined Vision Encoders:**
  * Evaluate standard backbones at 490x490 native resolution.
  * Contrastive vision encoders retain their dominance, establishing a clean baseline for text retrieval.

[TABLE: Re-baselined Retrieval Recalls on 2,000 pages setting (CLIP, SigLIP, DINOv2, MAE Reader, and Perfect-Text Upper Bound)]

---

# Slide 8: Main Focus: Visual Document-Pretrained Encoders (5:00 - 5:45)
## Evaluating Document AI Models Zero-Shot
* **The Document-Pretrained Family:**
  * Evaluate visual encoders explicitly pretrained on documents or document OCR tasks.
* **Nougat (Swin Transformer):**
  * Zero-shot visual page retrieval evaluation.
  * Shows that even models trained on document layouts require explicit retrieval adaptation.
* **Pix2Struct (Google):**
  * Process document patches by bypassing standard 224x224 image compression.
  * Evaluate zero-shot retrieval capabilities to set a baseline for document-pretrained models.

[TABLE: Zero-shot visual page retrieval recalls for Document-Pretrained Encoders (Nougat, Pix2Struct)]

---

# Slide 9: Main Focus: Cross-Modal Pretraining - I-JEPA with Text Targets (5:45 - 6:45)
## Predicting Language from Unmasked Image Context
* **Input & Target Flow:**
  * **Frozen BERT Text Encoder:** Takes tokenized ground-truth body text of the screenshot page crop as input. Outputs semantic text embeddings `(B, T, 768)`. Completely frozen.
  * **I-JEPA Context Visual Encoder:** Takes only unmasked visual patches of the screenshot crop. Processes via ViT-H/14 backbone (unfrozen/trainable during pretraining).
  * **Transformer Predictor:** Takes concatenated context visual tokens (with 2D positional embeddings) and target query tokens (learnable mask tokens + 1D text positional embeddings). Fully trainable.
* **Pretraining Objective:** Predict the semantic BERT embeddings of the page text from only the unmasked visual context.
* **Significance:** Forces the vision encoder to learn semantic text structures directly from visual layout and context, laying the groundwork for native-resolution document retrieval.

[FIGURE: Cross-Modal I-JEPA Text-Target Pretraining architecture diagram]


---

# Slide 10: Main Focus: Ongoing Experiment - Native-Resolution Training Grid (6:45 - 7:30)
## Probing High-Resolution Latent Emergence on Cluster
* **The Goal:** Benchmark standard image-only pretraining against our cross-modal text-target pretraining, and evaluate retrieval head configurations.
* **The Training Grid:**
  * Run parallel training runs using native 490x490 resolution (1,225 tokens per crop).
* **Experimental Configuration Matrix:**
  * **Image-Only Backbone:** Frozen/unfrozen I-JEPA backbone pretrained on images, comparing MLP vs. SALAD retrieval heads.
  * **Text-Target Backbone:** Unfrozen cross-modal I-JEPA backbone, comparing MLP vs. SALAD retrieval heads.
  * *Target:* Validate if text-target pretraining provides a stronger starting point for visual retrieval than pure image pretraining.

[TABLE: Experimental configuration matrix for the native resolution I-JEPA training runs]

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

[FIGURE: Visual illustration of random title whiteout masking during fine-tuning]

---

# Slide 12: Main Focus: Advanced Retrieval Architectures (8:15 - 8:45)
## Multi-Vector & Multi-Scale Vision
* **Two-Stream Global-Local Reader:**
  * A two-stream visual pipeline designed to solve the trade-off between local legibility and global layout context.
  * *Stream 1:* High-resolution native crops (490x490) for reading text.
  * *Stream 2:* Heavily downsampled global overview (224x224) to maintain layout structure.
* **Late Interaction (ColPali):**
  * Evaluate state-of-the-art multi-vector models (ColPali) using MaxSim scoring.
  * Test if single-vector models (SALAD) can reach multi-vector performance.

[FIGURE: Two-Stream Global-Local Reader architecture diagram]

---

# Slide 13: Future Research Agenda: Open Questions (8:45 - 9:30)
## Agenda for the June 14 Final Report
* **Q1. Feature Interpretability (SAE analysis):** Do features in SALAD/CLIP represent linguistic tokens (words/semantics) or only layout fingerprints (margins/headers)? Tested via Sparse Autoencoders.
* **Q2. Platonic Alignment on Visual Text:** Does a pixel-only model align linearly with text-only models (BERT) when trained exclusively on text screens?
* **Q3. Reading vs. Understanding:** Does the model perform basic OCR glyph-matching or higher-level semantic understanding? Evaluated via typographic attacks and perturbations.
* **Q4. Learning Dynamics:** At what point during fine-tuning does the model transition from learning layout coordinates to learning to read text?

---

# Slide 14: Conclusion & Takeaways (9:30 - 10:00)
## Summary of Work
* **Phase 1 Recap:** Pretraining objective determines zero-shot retrieval. Contrastive dominates; image-only has latent text structure but exploits layout coordinates when fine-tuned.
* **Phase 2 Contribution:** Resolution is the reading bottleneck. We introduced the legible 490x490 cropping protocol to enable true reading.
* **Ongoing Work:** Re-baselined standard encoders (Nougat, MAE), and launched parallel GPU runs to probe I-JEPA at native resolution.
* **Future Vision:** Combining title masking (to block layout shortcuts) with two-stream networks and late interaction (ColPali) to build robust OCR-free page search.
