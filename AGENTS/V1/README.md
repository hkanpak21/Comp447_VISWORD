# V1 — One-pager: what we're doing now (plain English)

This folder (`AGENTS/V1/`) is one unit of work: a spec ([PRD1.md](PRD1.md)), its task
tickets ([issues/](issues/)), and an append-only results ledger ([RESULTS.md](RESULTS.md)).
This page explains the whole thing so you don't have to reconstruct it from context.

---

## The project in two sentences

We feed a picture of a Wikipedia page to a vision model and ask it to retrieve the
right page. The scientific question: **do vision models actually *read* the text in
the image — and if so, how, and can they do it cheaply?**

---

## What we have right now

**The paper ("Can Vision Models Read?").** Its findings, all at 224×224 input:
- Models trained on image+caption pairs (CLIP, SigLIP) retrieve well from a crop
  (CLIP gets the right page in its top-10 about 78% of the time).
- Every model trained on images *only* (DINOv2, I-JEPA, an ImageNet classifier, a
  random network) is near chance (~2–6%). **I-JEPA is actually *below* a random
  network.**
- How well a vision model's "mental map" lines up with a text model's predicts its
  retrieval score (a correlation of 0.83) — i.e. reading ability tracks text-alignment.
- Fine-tuning DINOv2 on 50k pages reaches ~92%, beating zero-shot CLIP — **but it
  cheats**: it identifies pages by the *title bar* region, not by reading the body
  (painting out the top of the page makes its score collapse).

**What "JEPA" means here, and its exact state.** JEPA = "Joint-Embedding Predictive
Architecture": you hide part of an image and train the model to *predict the hidden
part's meaning* (not its pixels). The repo has two flavours:
1. **Plain I-JEPA** — predict the hidden image region's own features. Image-only.
2. **Text-Target I-JEPA** *(the interesting one)* — hide part of the screenshot and
   make the vision model predict a **language model's (BERT's) representation of the
   page's text**. In effect: teach the eyes to predict what the words mean.

Status of the JEPA work (updated after Barış's 6 June push): the **full Text-Target
training is finished** — there is a trained checkpoint (2026-06-03), and on its own it
retrieves at ~3% (still near chance, so the encoder alone isn't enough). Barış is now
**fine-tuning it with retrieval heads** (a simple MLP and the SALAD aggregator) — those
numbers are pending. Two catches: (1) the checkpoint lives in **Barış's private cluster
folder we can't read**, so he must share it or we re-run it; (2) the target so far is
the page **title** — we will switch it to the page **body**, which is the whole point of
"read all the text." A separate cheap probe also showed the reading signal is *latent*:
a small layer mapping frozen features to the text model reaches ~20% for I-JEPA (and
~72% for CLIP, which was already built to align with text).

**The data.** 376,000 Wikipedia page screenshots (each 980×980 pixels), already on the
cluster, plus a small set of "same page, different view" triplets for a transfer test.

**The compute.** A university cluster (Valar) with big-memory GPUs available, but it
runs ~one job at a time, and its compute nodes have no internet (so models/data must
be downloaded ahead of time).

---

## The one problem that limits everything

Each page is cut into tiles, and **each tile is shrunk 2.19× before the model sees
it.** That shrink turns body text into an unreadable blur. So when the paper says a
model "reads," it can only mean it reads *headlines and layout* — never the body. Every
"reading" claim, and the title-cheating failure, trace back to this single shrink.

---

## What we'll do — step by step

**0. Protect what exists, then prepare.** Barış's newest paper edits live only on the
cluster (not on GitHub) and aren't safely anchored — first we put them on a branch so
they can't be lost. Then back up the data, results, and paper to permanent storage, and
download the new reading-models for offline use. *(Nothing is deleted or overwritten;
any git operation waits for your OK.)*

**1. Stop shrinking the text.** Re-cut pages into full-resolution tiles, and add
smart cropping so each tile is a clean unit of text (no blank tiles, no half-cut
lines). Now the body is legible to *every* model.

**2. Re-run the science under legibility.** Re-measure all six original models, the
text-alignment correlation, and the title-cheating test — now that text is readable.
*Does the old ranking still hold when models can actually see the words?*

**3. Bring in models already trained to read.** Add four off-the-shelf models built to
read rendered text/documents (Pix2Struct, Donut, Nougat, ColPali). Expectation: they
top the chart — a clean new result about *which kind of pretraining produces reading*.

**4. Build the efficient reader (the JEPA contribution) — on top of Barış's work.**
He already has the trained encoder + the retrieval-head code; we (a) get his checkpoint
(shared or re-run), (b) finish the head results he left pending, then (c) make our
addition: switch the target from the title to the **body** text and evaluate at the
**page** level. Test the same model before vs after so the gain is clearly from the
method. Then add a "if reading were perfect" reference line using the real text.

**5. Show and check.** Produce heatmaps of *where* each model looks (does it look at
the body now?), and confirm the title-cheating is gone once text is legible.

**Later (separate folders):** harder documents (arXiv papers) → V2; question-answering
about pages → V3. Only after V1 looks good.

---

## What we get at the end of V1

A legible dataset; a fair, re-run comparison of how different pretraining reads (now
including purpose-built reading models); a cheap JEPA reader that demonstrably learns
to read the body and beats the old 20% probe; "where it reads" heatmaps; and proof the
title-cheating is fixed — all written into the next version of the report, on top of
Barış's branch. Every number goes into [RESULTS.md](RESULTS.md), **accumulated, never
overwritten.**
