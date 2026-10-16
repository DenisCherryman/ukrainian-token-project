# UTP v0.8 — append Ukrainian pieces, preserve the original pretrained vocabulary

**Engineering pilot with a very small CC0 corpus, not a deployment result.**

Original model: HuggingFaceTB/SmolLM2-135M at 93efa2f097d58c2a74874c7e644dbc9b0cee75a2.
Old tokens: 49152; appended: 128; total: 49280.
English strings with identical IDs: 2000.

| Condition | UA tokens | UA bits/UTF-8 byte | EN tokens | EN bits/UTF-8 byte |
|---|---:|---:|---:|---:|
| Original tokenizer + pretrained model | 880 | 2.0112 | 560 | 1.2538 |
| Appended Ukrainian tokens; 0 new-row train steps | 819 | 2.0539 | 560 | 1.2538 |
| Appended Ukrainian tokens; 64 new-row train steps | 819 | 1.9567 | 560 | 1.2538 |
| Appended Ukrainian tokens; 256 new-row train steps | 819 | 1.8942 | 560 | 1.2539 |

## Limitations

- Exploratory engineering test; Tatoeba source previously evaluated in UTP v0.5-v0.7.
- Only new rows were updated; other transformer and original embedding weights frozen.
- Adding output vocabulary rows changes softmax even when old input IDs and embeddings are unchanged.
- 30 sentences per language, same source, no independent quality/domain/task test.
- Tokenizer pieces selected on CC0 TRAIN only, not on heldout token counts.
- Training checkpoints are repeated looks at same heldout sample; no checkpoint selected by results.
- GitHub CPU runtime is not a benchmark of production latency.
- No relation to ChatGPT or GPT-6 token cost; no model weights or corpora redistributed.
