# UTP v0.5 — independent Tatoeba CC0 test

**Out-of-domain tokenization test, NOT deployed-model savings.**

| Language | CC0 sentences | Exact UD overlaps removed |
|---|---:|---:|
| uk | 393 | 0 |
| en | 2000 | 0 |

| Seed | 32K selected by UD DEV | UA CC0 token reduction | EN CC0 token reduction |
|---:|---|---:|---:|
| 42 | uk75 | +4.904% | -6.931% |
| 173 | uk75 | +5.153% | -6.514% |
| 2026 | uk75 | +3.745% | -6.478% |

Descriptive mean across subset seeds: UA +4.601%, EN -6.641%.

## Cautions

- Independent new SOURCE, not independent linguistically diverse documents: sentence authorship and near-duplicate sources are unknown.
- Exact UD sentence overlap excluded; paraphrase and training-history leakage cannot be ruled out.
- Only explicitly CC0-labeled Tatoeba sentences evaluated. The training data are noncommercial UD and learned weights are not redistributed.
- Evaluation sentence counts may be small. These seed runs reuse the same independent CC0 test and are not independent corpus replicates.
- No confidence intervals: individual sentences may be clustered by author or source and independence cannot be established.
- No GPT-6 token savings, model quality, or API pricing conclusion can be drawn.
