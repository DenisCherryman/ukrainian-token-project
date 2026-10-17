# UTP v0.9 — preregistered external evaluation

Goal: evaluate the frozen v0.8 additive tokenizer on Ukrainian and English encyclopedia articles as a genuinely different text genre. Do not train on, select tokens from, or tune against the external holdout. Preserve original token IDs. Collect source page titles, revision IDs and CC BY-SA 4.0 attribution, but do not commit the article text. Compare identical UTF-8 text before and after adaptation, and report both token counts and bits per original UTF-8 byte. Reject partial or unpaired samples and disclose the possibility that a pretrained model has seen Wikipedia during pretraining. Repeat with three fixed training seeds before any generalization claim.


## Fixed evaluation rules

Use 30 different Ukrainian and 30 different English encyclopedia articles, one sentence per article, only for evaluation. Record article URLs, revision IDs, timestamps, source license, and sentence SHA-256. Exclude exact and near duplicates with the training set. Require identical UTF-8 bytes in all model conditions. Train with seeds 42, 173, and 2026. Evaluate original, 0, 64, and 256 adaptation steps without selecting checkpoints on test. English bits/UTF-8 byte regression over 1% counts as failure. Do not redistribute article texts or model weights. Wikipedia may have been seen during original model pretraining.
