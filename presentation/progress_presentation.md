Slide 1: Can Vision Models Read?1
Can Vision Models Read?1
Probing Linguistic Structure via JEPA on Text Screenshots1
Barış Cem Bakay, Halil İbrahim Kanpak1
Department of Computer Engineering, Koç University1
Slide 2: Motivation1
Documents combine glyphs (text as visual patterns) and layout (titles, infoboxes, columns, figures).1
A vision encoder fed a page screenshot could in principle learn both. We do not know which models actually do, and to what extent.1
Standard text-pipeline retrieval (OCR + text encoder) discards layout, fails on degraded scans, and cannot accept image queries.1
Visual-pipeline retrieval bypasses these limits but places the burden of "reading" on the vision encoder.1
Image Description: system figure1
Slide 3: Research questions1
Q1. Which pretraining produces image representations from which the source page can be retrieved zero-shot?1
Q2. Does I-JEPA, image-only predictive pretraining with no text supervision, acquire any linguistic structure detectable through retrieval?1
Q3. Does the Platonic Representation Hypothesis (Huh et al., ICML 2024) predict the retrieval ordering across encoders quantitatively?1
Slide 4: Background1
Pixel-based language models: PIXEL, CLIPPO, Pix2Struct train transformers directly on rendered text.1
Visual document retrieval: ColPali (Faysse et al., 2025) shows multi-vector page embeddings competing with OCR pipelines.1
Self-supervised vision: DINOv2 (view contrastive distillation), I-JEPA (masked latent prediction). Jose et al. 2024 align a small text encoder post-hoc to frozen DINOv2 to recover CLIP-level zero-shot classification.1
Image-text contrastive: CLIP (softmax InfoNCE), SigLIP (sigmoid pairwise). Both have been shown to read text rendered in images.1
Platonic Representation Hypothesis: encoders trained with different objectives converge to a shared abstract geometry as scale grows.1
Slide 5: Training protocol1
Backbone: ViT-B family, last 4 of 12 blocks unfrozen.1
Head: MLP (768 -> 512 -> 256) or SALAD optimal-transport aggregator (8448-d).1
Loss: multi-similarity. Batch = 4 pages x 4 crops = 16 images.1
Optimiser: AdamW, weight decay 1e-4, gradient clip 1.0, linear warmup over 5% of steps then cosine decay.1
Backbone LR: 1e-5 for DINOv2 / image-only, 1e-7 for CLIP and SigLIP. Head LR: 5e-4.1
30,000 training pages x 3 epochs. One seed.1
Slide 6: Evaluation protocol1
We evaluate retrieval over a held-out 2,000-page slice of Wikipedia screenshots.1
Setting A: gallery is one vector per page (mean over that page's crops). Query is a single 224x224 crop. Random R@1 = 5e-4.1
Setting B: gallery is the same page's other crops in a small pool with hard negatives from unrelated pages. Tests cross-set transfer to a different crop distribution.1
Slide 7: Zero-shot retrieval (Setting A, P=2000)1

Image-text contrastive encoders dominate by ~0.7 R@10. I-JEPA scores below random init despite being the largest model.Slide 8: Platonic alignment predicts retrieval1
For each (image encoder, text encoder) pair we compute mutual-kNN@10 alignment on the same 2,000 pages.1
Image-text contrastive encoders align with text encoders at 0.09-0.14 mutual-kNN@10. Image-only encoders align at 0.007-0.027.1
Across the 6 encoders: Spearman rho = 0.83, p = 0.04 between text-alignment and zero-shot R@10.1
[Citation] Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). Position: The Platonic Representation Hypothesis. In Proceedings of the 41st International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, 235, 20617–20642.1
Slide 9: Fine-tuning closes (and overshoots) the gap1

DINOv2-SALAD-50k exceeds zero-shot CLIP. CLIP fine-tuning at conservative LR=1e-7 hurts more than it helps.Slide 10: What pixels are the encoders using?1
We re-encode the eval slice with the top 15% of every page painted white, then re-rank.1
Zero-shot CLIP: R@10 0.779 -> 0.852 (+0.073). SigLIP: +0.106. Image-only encoders: flat. Falsifies "CLIP just reads titles."1
Fine-tuned DINOv2-SALAD-50k: R@10 0.915 -> 0.673 (-0.242). CLIP-SALAD-30k: -0.037. Fine-tuning has taught DINOv2 to bind page identity to the title region.1
On Setting B (cross-set), the same DINOv2 checkpoints gain up to 6.7x R@1 from blanking. The in-distribution shortcut is the cross-set confound.1
Slide 11: I-JEPA has latent text alignment1
I-JEPA features are not zero-shot text-aligned (image-text R@10 = 0.015 against BERT[CLS] of titles).1
Train a single linear projection A: 1280 -> 768 from I-JEPA into BERT space on 4,000 (screenshot, title) pairs by InfoNCE.1
Adapted I-JEPA <-> BERT R@10 = 0.197 (13x improvement, no nonlinearity, no encoder finetuning).1
Image-only predictive pretraining encodes text-aligned structure that is invisible in the wrong basis.1
Slide 12: Late interaction is not the bottleneck-breaker1

Late interaction marginally improves R@10 but loses R@1 by 4x on CLIP. DINOv2 stays at chance regardless of aggregation. The encoder is the bottleneck, not the aggregator.Slide 13: LEACE causal title-erasure1
We isolate the "title direction" in feature space: stack the per-crop deltas phi(orig) - phi(blank), take the top-1 PCA axis.1
LEACE projects this axis out of every feature.1
CLIP R@10: 0.779 (orig) -> 0.823 (blanking) -> 0.863 (LEACE). Erasing one feature-space axis raises retrieval more than the pixel intervention.1
DINOv2 R@10: 0.058 -> 0.056 -> 0.068. The title direction barely exists as a structured axis in DINOv2.1
Slide 14: Discussion1
Pretraining objective is the dominant predictor of zero-shot document retrieval. Image-text contrastive wins by a wide margin.1
The Platonic axis explains the ordering: text-alignment predicts retrieval at rho = 0.83.1
Fine-tuning can rescue image-only encoders past zero-shot CLIP, but does so by overfitting to a single layout region (the title).1
The "title direction" in CLIP feature space is a real linear axis; erasing it surgically beats removing the pixels.1
I-JEPA encodes latent text alignment, but only in the wrong basis; a thin linear adapter recovers a substantial fraction.1
Aggregation choice (single-vector vs late interaction) matters much less than encoder choice.1
Slide 15: Limitations1
One seed per fine-tune. Within-family comparisons need multi-seed reruns.1
Eval pool P = 2,000. Industrial corpora are 1e5 - 1e6 pages.1
Crop input size 224 x 224 makes body text sub-pixel; "reading" here means heading-level text and visual layout, not fluent body.1
I-JEPA adapter R@10 = 0.197 on a 1,000-page set is well below CLIP zero-shot on the larger pool; the result shows latent alignment exists, not that it is practical.1
Slide 16: Conclusion1
Image-text contrastive pretraining dominates zero-shot visual document retrieval. Image-only baselines sit near chance.1
Alignment with text-encoder geometry predicts retrieval performance: the Platonic Hypothesis holds in this domain.1
Fine-tuning can exceed zero-shot CLIP but introduces a universal title-region shortcut, visible both as a Setting-A drop and a Setting-B gain under blanking.1
I-JEPA carries latent text-aligned structure recoverable by a single linear projection.1
The encoder, not the aggregator, is the limiting factor.
Slide 17: Phase 2 Progress - Legible Native Resolution1
We identified that previous evaluations scaled 490x490 crops down to 224x224, destroying body text legibility (2.19x shrink).1
We implemented a `TextAwareCropper` that processes crops at native resolution (490x490) with line-snapping heuristics.1
Result: The model now receives perfectly legible, text-aware tiles without splitting lines horizontally.1

Slide 18: Re-baselining the Encoder Grid1
We evaluated all baseline encoders (CLIP, SigLIP, DINOv2, etc.) at native legible resolution over 2,000 pages.1
We also added a "Perfect-Text Upper Bound": Mean-pooling BERT embeddings directly from the ground-truth text of the page yields R@10 = 0.938.1
We introduced a Contrastive MAE Reader (our custom text-targeted training on MAE features) as a strong baseline, reaching R@10 = 0.098.1

Slide 19: The Document-Pretrained Family1
To contextualize our models, we evaluated OCR-free encoders explicitly trained to read document images.1
We evaluated Nougat (Swin Transformer, R@10=0.049) and Pix2Struct (running).1
This establishes a ceiling for how well off-the-shelf document understanding models handle our retrieval task without explicit fine-tuning.1

Slide 20: Ongoing Experiment: I-JEPA at Native Resolution1
Hypothesis: I-JEPA features naturally capture text structure if we pretrain with a body-text target and evaluate at native resolution.1
We launched 4 extensive training runs on A40 GPUs using 490x490 resolution (1225 tokens per crop).1
We are comparing Image-Only vs. Text-Target (Body) pretraining, paired with MLP vs. SALAD retrieval heads.1
Results will conclusively show if predicting hidden states (I-JEPA) can induce semantic reading capabilities.1

Slide 21: Next Steps & Future Directions1
Confound Control (Title Masking): Applying a random whiteout mask to the title during training to force the model to read the body text, preventing the "layout cheat."1
Global Page Context: A two-stream reader combining high-res 490x490 tiles with a heavily downsampled 224x224 overview of the entire page to retain global layout.1
Late Interaction (ColPali): Evaluating state-of-the-art multi-vector models (ColPali) using MaxSim scoring to see if single-vector models can match them.1
