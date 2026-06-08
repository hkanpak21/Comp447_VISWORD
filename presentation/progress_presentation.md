# Slide 1: Can Vision Models Read? (0:00 - 0:45)
## Can Vision Models Read?
### Probing Linguistic Structure via JEPA on Text Screenshots
* **Authors:** Barış Cem Bakay, Halil İbrahim Kanpak
* **Affiliation:** Department of Computer Engineering, Koç University
* **Goal:** Investigate whether self-supervised and contrastive vision encoders learn linguistic structure and page layout directly from document images.

---

# Slide 2: Motivation (0:45 - 1:30)
## Visual Document Retrieval vs. Text Pipelines
* **The OCR Bottleneck:** Standard document retrieval relies on OCR + text encoders, discarding layout cues (headers, columns, tables), failing on degraded scans, and ignoring image queries.
* **OCR-Free Visual Retrieval:** Processing page screenshots directly preserves layout and visual elements, but shifts the entire burden of "reading" to the vision encoder.
* **Pretraining Paradigms:**
  * *Image-Text Contrastive (CLIP, SigLIP):* Trained on web-scale image-caption pairs; known to read rendered text.
  * *Image-Only Self-Supervised (DINOv2, I-JEPA):* No text supervision. Do they emerge with linguistic structure?

---

# Slide 3: Research Questions (1:30 - 2:15)
## Core Inquiries
* **Q1. Zero-Shot Capability:** Which vision pretraining enables a model to retrieve the source page of a crop zero-shot?
* **Q2. The Platonic Representation Hypothesis:** Does semantic text alignment correlate with retrieval performance across different encoders?
* **Q3. Latent Structure in Image-Only Models:** Can predictive pretraining (I-JEPA) encode linguistic structures that are invisible without a change of basis?
* **Q4. The Resolution Confound:** Does standard 224x224 resolution limit models to layout matching, and can native-resolution processing unlock true reading?

---

# Slide 4: Experimental Setup & Protocols (2:15 - 3:00)
## Evaluation Protocols A & B
* **Evaluation Slice:** Held-out 2,000-page slice of Wikipedia screenshots.
* **Protocol A (Page Retrieval):** 
  * Gallery: One vector per page (mean of page crops).
  * Query: A single page crop. Evaluates page re-identification.
* **Protocol B (Cross-Set Transfer):**
  * Gallery: Other crops of the same page in a pool with hard negatives.
  * Evaluates transferability to different layout regions.
* **Aggregation Heads:** Single-vector pooling, MLP (768 $\rightarrow$ 512 $\rightarrow$ 256), and SALAD optimal-transport aggregator (8,448-d).

---

# Slide 5: The Legibility Bottleneck: 224 vs. 490 (3:00 - 3:45)
## Restoring Native Resolution
* **The Squashing Problem:** Phase 1 scaled 490x490 crops down to 224x224, shrinking characters by 2.19x. This rendered body text sub-pixel (illegible). 
* **The Shortcut:** Under this bottleneck, models were forced to rely on global page layout or prominent headings, rather than reading fluent body text.
* **Solution: Text-Aware Cropper:**
  * Processes page crops at native 490x490 resolution (1,225 tokens per crop).
  * Implements line-snapping heuristics to keep lines intact.
  * Guarantees perfectly legible text inputs to the vision encoders.

---

# Slide 6: Re-Baselining & Bounds at Native Resolution (3:45 - 4:45)
## Results & Performance Upper Bounds
* **Perfect-Text Upper Bound (BERT):** Mean-pooling BERT embeddings directly from the ground-truth text of the page yields **R@10 = 0.938**. This represents the text-only retrieval ceiling.
* **Contrastive MAE Reader:** A strong custom baseline trained directly on MAE features to reconstruct/embed text, achieving **R@10 = 0.098**.
* **Zero-Shot Baselines:**
  * Contrastive models (CLIP, SigLIP) dominate zero-shot retrieval.
  * Image-only models (DINOv2, I-JEPA) sit near chance levels zero-shot.
  * **Document-Pretrained Family:** Swin-based Nougat gets **R@10 = 0.049**, showing that document pretraining alone does not guarantee zero-shot retrieval capability.

---

# Slide 7: The Platonic Representation Hypothesis (4:45 - 5:30)
## Alignment Predicts Retrieval
* **Mutual-kNN@10 Alignment:** Measures the spatial alignment between image encoders and a text encoder (BERT) on the same 2,000 pages.
* **Alignment Scores:**
  * Contrastive Encoders: Align with BERT at **0.09 - 0.14** mutual-kNN@10.
  * Image-Only Encoders: Align poorly at **0.007 - 0.027**.
* **Quantitative Correlation:**
  * Spearman's $\rho = 0.83$ ($p = 0.04$) between text-alignment and zero-shot retrieval R@10.
  * *Implication:* The Platonic Hypothesis holds. Models that learn to represent the semantic structure of text are those that excel at visual document retrieval.

---

# Slide 8: Fine-Tuning & The Layout Confound (5:30 - 6:30)
## Shortcut Learning vs. Invariant Reading
* **The Fine-Tuning Paradox:** Fine-tuning DINOv2-SALAD-50k yields **R@10 = 0.915**, exceeding zero-shot CLIP.
* **Title-Blanking Experiment:** Painting the top 15% (title region) of the page white during evaluation:
  * Fine-Tuned DINOv2: Crashes by **-0.242 R@10** (0.915 $\rightarrow$ 0.673).
  * Zero-Shot CLIP: Gains **+0.073 R@10** (0.779 $\rightarrow$ 0.852).
* **The Layout Cheat:** Fine-tuning teaches image-only models to bind page identity to spatial layout shortcuts (e.g., titles). Contrastive models (CLIP/SigLIP) focus on translation-invariant text patterns.

---

# Slide 9: LEACE Causal Title-Erasure (6:30 - 7:30)
## Surgically Erasing the Shortcut
* **Isolating the Title Axis:** We calculate the difference vector $\phi(orig) - \phi(blank)$ in feature space and extract the top PCA axis.
* **Causal Projection (LEACE):** We project this "title direction" out of the features.
* **Retrieval Recall (R@10) Comparison:**
  * *CLIP:* 0.779 (Original) $\rightarrow$ 0.823 (Blanking pixels) $\rightarrow$ **0.863** (LEACE Erased).
  * *DINOv2:* 0.058 $\rightarrow$ 0.056 $\rightarrow$ **0.068**.
* **Interpretation:** 
  * The title exists as a structured, linear axis in CLIP's representation. Erasing it surgically is more effective than removing the actual pixels.
  * In DINOv2, the title direction is not linearly structured, confirming it does not represent titles as distinct text concepts.

---

# Slide 10: I-JEPA: Latent Alignment & Native Training (7:30 - 8:45)
## Unlocking Predictive Pretraining
* **Latent Text Alignment:** 
  * I-JEPA features show poor zero-shot alignment (R@10 = 0.015).
  * Training a single linear projection (InfoNCE on 4,000 screenshot-title pairs) into BERT space yields **R@10 = 0.197** (a 13x boost).
  * *Finding:* Predictive pretraining encodes text structure, but in a different basis.
* **Ongoing Native-Resolution I-JEPA Training:**
  * Training I-JEPA at native 490x490 resolution (1,225 tokens) on 4 A40 GPUs.
  * Comparing *Image-Only* vs. *Text-Target (Body)* pretraining objectives.
  * Evaluating *MLP* vs. *SALAD* retrieval heads.
  * Determining if predicting masked states can emerge with semantic reading capabilities.

---

# Slide 11: Next Steps & Future Architectures (8:45 - 9:30)
## Road Map
* **Confound Control (Title Masking):** Randomly masking title regions during training to force the model to read the body text, preventing layout-based shortcut learning.
* **Two-Stream Global Context:** Combining a high-res native crop (490x490) for reading with a low-res global overview (224x224) to maintain layout awareness.
* **Late Interaction Evaluation:** Comparing single-vector retrieval against state-of-the-art multi-vector models (like ColPali) using MaxSim scoring.

---

# Slide 12: Conclusion (9:30 - 10:00)
## Key Takeaways
* **Pretraining Objective Matters:** Image-text contrastive learning is crucial for zero-shot text retrieval. Image-only models sit near chance.
* **The Platonic Correlation:** Vision-text representation alignment is a strong predictor of document retrieval success.
* **Resolution is Key:** Processing at native 490x490 resolution is necessary to support actual reading over layout-based shortcut matching.
* **Latent Semantics Exist:** Image-only predictive models (I-JEPA) encode linguistic structure, recoverable via a linear adapter, prompting our native resolution I-JEPA training grid.
