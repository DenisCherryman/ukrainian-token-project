# UTP v0.7 — bilingual embedding continuation, open-weight LLM

**Exploratory engineering test. Lower bits/UTF-8 byte = better prediction.**

Model: HuggingFaceTB/SmolLM2-135M (93efa2f097d58c2a74874c7e644dbc9b0cee75a2)

| Condition | Ukrainian bits/byte | UA text tokens | English bits/byte | EN text tokens |
|---|---:|---:|---:|---:|
| Original pretrained tokenizer | 2.0111 | 880 | 1.415 | 357 |
| New 1024 vocabulary; 0 embedding steps | 2.4566 | 514 | 4.4317 | 820 |
| New 1024 vocabulary; 16 embedding steps | 2.2891 | 514 | 4.3632 | 820 |
| New 1024 vocabulary; 64 embedding steps | 2.1819 | 514 | 4.1215 | 820 |
| New 1024 vocabulary; 128 embedding steps | 2.0663 | 514 | 3.9792 | 820 |

## Caveats

- Engineering smoke test using a small CC0 dataset; no newly independent domain or downstream task test.
- The same source (Tatoeba) was used in v0.5 and v0.6 exploratory work.
- Checkpoints are multiple looks at the same heldout texts, not independent confirmatory evaluations.
- Only embedding tables were updated; the transformer backbone remained frozen.
- The 1024-token new vocabulary differs in size from the original model tokenizer.
- No licensed/restricted UD text or UD-trained model is used in this workflow.
- Training corpus is too small to demonstrate model-quality recovery or generalizable improvements.
- Raw forward pass time is not hardware-normalized production speed.
- No ChatGPT/GPT-6 token-price or quality improvement implied.
