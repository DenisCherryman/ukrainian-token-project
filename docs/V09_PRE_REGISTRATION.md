# UTP v0.9 — preregistered external evaluation

Goal: evaluate the frozen v0.8 additive tokenizer on Ukrainian and English encyclopedia articles as a genuinely different text genre. Do not train on, select tokens from, or tune against the external holdout. Preserve original token IDs. Collect source page titles, revision IDs and CC BY-SA 4.0 attribution, but do not commit the article text. Compare identical UTF-8 text before and after adaptation, and report both token counts and bits per original UTF-8 byte. Reject partial or unpaired samples and disclose the possibility that a pretrained model has seen Wikipedia during pretraining. Repeat with three fixed training seeds before any generalization claim.


## Fixed evaluation rules

Use 30 different Ukrainian and 30 different English encyclopedia articles, one sentence per article, only for evaluation. Record article URLs, revision IDs, timestamps, source license, and sentence SHA-256. Exclude exact and near duplicates with the training set. Require identical UTF-8 bytes in all model conditions. Train with seeds 42, 173, and 2026. Evaluate original, 0, 64, and 256 adaptation steps without selecting checkpoints on test. English bits/UTF-8 byte regression over 1% counts as failure. Do not redistribute article texts or model weights. Wikipedia may have been seen during original model pretraining.


## Audit and deterministic protocol (2026-10-17, before any v0.9 evaluation)

**What "independent" means here:** the encyclopedia holdout is external to
the Tatoeba CC0 tokenizer-selection/adaptation data. It is **not guaranteed**
independent of SmolLM2 pretraining; the original model may have seen Wikipedia.
The holdout is for a *domain-transfer pilot*, not a proof of generalization to
unseen training content. No v0.9 scores existed when this protocol was added.

### Data acquisition and deterministic sentence choice

1. Fix the actual source snapshot/revision before measuring model scores.
   Record language, Wikipedia article page ID, exact revision ID, timestamp,
   article URL, retrieval date, source license, extraction-script commit, and
   SHA-256 of the selected original UTF-8 sentence. Keep a separate private
   transient working cache if needed; **do not commit sentence text**.
2. From main-namespace, non-redirect pages, sort candidate page IDs by
   SHA-256 of the UTF-8 string UTP-v09|<language>|<page_id> ascending.
   This prevents manually selecting pages that favor the candidate tokenizer.
   Pin and record the page universe/snapshot hash; otherwise this ranking is
   not independently reproducible.
3. On each pinned revision, remove markup, templates, tables, citations and
   footnotes using one version-pinned extractor. Choose the **first** complete
   prose sentence of 60–300 Unicode code points (not first *favorable*
   sentence); skip a page if none exists. Preserve the exact selected text
   unchanged in the transient evaluation cache. Record the extraction rule
   and tool version in the manifest.
4. Exclude sentences identical or near-duplicate to any available Tatoeba
   TRAIN or earlier test example after Unicode NFC, casefold and whitespace
   normalization. Also reject duplicate selected sentences and multiple
   selections from the same page. Publish the overlap-check method and
   number of excluded pages, not excluded texts.
5. Pre-filter **before scoring**, requiring the exact same 30 selected UA
   and 30 EN sentences to fit both original and augmented context limits.
   If fewer than 30 are available, fail rather than silently substitute or
   skip sentences during evaluation. Do not change the selection after seeing
   quality metrics.
6. Keep model and tokenizer revision pinned; record exact dependencies,
   CPU architecture, deterministic seeds 42/173/2026 and per-seed model
   configuration. The 128 Ukrainian pieces must be chosen from Tatoeba TRAIN
   for **each seed** or held fixed from the pre-registered v0.8 construction,
   with that choice declared before training; no selection on Wikipedia.
7. Record only source metadata, aggregate token counts, aggregate bits/UTF-8
   byte, per-seed deltas and document-cluster confidence intervals. Do not
   commit copyrighted article excerpts, token surfaces, model weights or
   any potentially restricted source corpus.

### Statistics and acceptance

- Primary paired outcome: difference in **total NLL bits / total original
  UTF-8 bytes**, recomputed from the *same document resamples* for original
  and adapted models. Resample article IDs with replacement (2,000 bootstrap
  draws, seed 2026), preserving paired bytes. **Do not use an unweighted
  mean of per-document bits/byte as a substitute**: that estimates a different
  quantity and gives short and long articles equal weight.
- Secondary outcome: change in aggregate text-token count on the same
  documents. Report absolute and relative changes and paired bootstrap CI.
  Report Ukrainian and English separately and all three training seeds,
  including unfavorable results. Never select a checkpoint or seed by test
  performance.
- English guardrail: >1% deterioration in bits/UTF-8 byte against the same
  baseline fails that seed; a zero token-count change alone is not evidence
  of equal quality. Report observed confidence intervals and small-sample
  uncertainty rather than treating a non-significant difference as proof
  of equivalence.
- No production savings or inference-speed claims until matched-hardware
  throughput, memory, latency and task-quality tests have been performed.

**Implementation status:** these are preregistered requirements, not evidence
that an external benchmark has already been run.
