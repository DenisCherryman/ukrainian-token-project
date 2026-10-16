# Ukrainian Token Project — real-corpus benchmark

**Non-commercial tokenizer-only experiment. NOT ChatGPT savings.**

Executed: 2026-10-16T22:05:28.167361+00:00

## Document-disjoint splits

| Language | Train (sentences/docs) | Dev | Test |
|---|---:|---:|---:|
| uk | 4931/475 | 549/64 | 1510/65 |
| en | 11984/968 | 1408/100 | 2013/106 |

## Frozen model selection (dev only)

Baseline UA share: 50%; selected: 75%.

## Held-out test results

- **uk_test: +7.782%** reduction; 95% CI [7.553, 8.266], bootstrap unit: document.
- **en_test: -9.462%** reduction; 95% CI [-9.957, -8.991], bootstrap unit: document.

## Limitations

- Tokenizer-only experiment, NOT API price/quality/latency or GPT-6 saving.
- UD_Ukrainian-IU is CC BY-NC-SA 4.0; do not redistribute source texts or learned model weights.
- Ukrainian and English datasets are not sentence-aligned translations.
- tiktoken vocabularies differ in size; references are not controlled comparisons.
- Exact sentence deduplication does not prevent paraphrase/topic leakage.
- Single small corpus cannot represent all Ukrainian language use.

See matching JSON for aggregate numbers and SHA-256 source checksums.
