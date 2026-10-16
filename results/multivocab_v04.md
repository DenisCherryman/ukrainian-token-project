# UTP v0.4 — vocabulary scaling on actual UD text

**Noncommercial tokenizer experiment; not ChatGPT savings.**

Requested vocabularies and independently sampled training subsets share the same held-out test set.

| Vocab requested | Seed | Actual vocab baseline/candidate | Selected on dev | UK test reduction | EN test reduction |
|---:|---:|---|---|---:|---:|
| 1024 | 42 | [1024, 1024] | uk75 | +7.782% | -9.462% |
| 1024 | 173 | [1024, 1024] | uk75 | +6.701% | -8.232% |
| 1024 | 2026 | [1024, 1024] | uk75 | +7.260% | -8.422% |
| 4096 | 42 | [4096, 4096] | uk75 | +6.768% | -8.424% |
| 4096 | 173 | [4096, 4096] | uk75 | +6.789% | -9.298% |
| 4096 | 2026 | [4096, 4096] | uk75 | +7.038% | -9.626% |
| 8192 | 42 | [8192, 8192] | uk75 | +6.088% | -7.996% |
| 8192 | 173 | [8192, 8192] | uk75 | +6.312% | -8.146% |
| 8192 | 2026 | [8192, 8192] | uk75 | +5.697% | -7.641% |
| 16384 | 42 | [16384, 16384] | uk75 | +4.799% | -7.702% |
| 16384 | 173 | [16384, 16384] | uk75 | +4.848% | -7.347% |
| 16384 | 2026 | [16384, 16384] | uk75 | +5.005% | -6.186% |
| 32768 | 42 | [32768, 32768] | uk75 | +4.695% | -7.603% |
| 32768 | 173 | [32768, 32768] | uk75 | +4.689% | -6.581% |
| 32768 | 2026 | [32768, 32768] | uk75 | +4.816% | -5.992% |

## Mean reductions across subset-selection seeds (descriptive)

| Vocab | Seeds evaluated | Ukrainian | English |
|---:|---:|---:|---:|
| 1024 | 3 | +7.248% | -8.705% |
| 4096 | 3 | +6.865% | -9.116% |
| 8192 | 3 | +6.032% | -7.928% |
| 16384 | 3 | +4.884% | -7.078% |
| 32768 | 3 | +4.733% | -6.725% |

## Limits

- Noncommercial dataset; source text and model artifacts not released.
- This is NOT ChatGPT or GPT-6 token, cost, speed or quality improvement.
- Seeds resample the same documents; their outcomes are NOT independent corpus replications.
- Test documents are shared across all configurations. Bootstrap CI does not account for domain shift.
- If a requested vocabulary size cannot be reached by BOTH models, mark it skipped.
