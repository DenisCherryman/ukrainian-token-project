# UTP v0.4: vocabulary-scale experiment

## Research question

Does increasing the fraction of Ukrainian in a **fixed-size** bilingual SentencePiece-BPE vocabulary consistently reduce token counts on previously unseen Ukrainian documents without unacceptable English regression? This is a **tokenizer-only** experiment, not a change to GPT models or API billing.

## Setup

- Source: pinned **UD Ukrainian-IU** (CC BY-NC-SA 4.0, noncommercial research) and **UD English-EWT** (original text rights may differ), using the same exact upstream revisions as [the first real-corpus experiment](../results/ud_iu_ewt_v03.md).
- Train/dev/test documents assigned once using the stable document-ID hash in prepare_ud.py. The same independent documents are shared across runs. Exact cross-split text duplicates are excluded.
- Requested vocabulary sizes: **1,024, 4,096, 8,192, 16,384 and 32,768** tokens. Models within each comparison must have the exact same effective vocabulary size. If the corpus cannot produce both requested sizes, that comparison is flagged **SKIPPED**, not counted as a result.
- Repeat training-subset sampling with seeds **42, 173, 2026**. Training mixtures: **50%** vs **75% Ukrainian by UTF-8 training bytes**. Maximum aggregate training budget: **500,000 bytes**, capped further by available train data. Each candidate and baseline uses the same cap.
- Selection happens **only on dev**. Use Ukrainian-heavy candidate only if Ukrainian tokens decrease and English token regression on dev is at most **10%**. Otherwise freeze the baseline choice. All choices are recorded *before* test evaluation.
- Final held-out test evaluates these frozen choices with paired document-cluster bootstrap (600 resamples) and validates **lossless text round-trips**. A separate held-out corpus and confidence-adjustment would be required for inferential comparisons across many explored settings.

## Running

Run on a machine with Python 3.12 and sufficient CPU/RAM:

~~~bash
pip install -r requirements.txt
pytest -q
python corpora.py --out data/ud_raw --accept-license \
  --uk-ref 21f641ba6109044ce9aafd4b2cbfa20a4ac6e479 \
  --en-ref 4a4d77f599ea53cc405f85d0cec4b2f14f81d42b
python prepare_ud.py --source data/ud_raw --out data/ud_strict
python multivocab.py --data data/ud_strict \
  --work runs/multivocab_v04 \
  --out results/multivocab_v04.json \
  --vocabs 1024 4096 8192 16384 32768 \
  --seeds 42 173 2026 --train-budget-bytes 500000 \
  --max-en-regression-pct 10 --bootstrap 600
~~~

A GitHub Actions workflow runs the same procedure: [ud_multivocab.yml](../.github/workflows/ud_multivocab.yml).

## Caveats and release boundaries

- Varying seed changes *subset order/selection*, not the evaluation documents. Three seeds are **not** three independent corpus replications.
- This comparison uses **equal vocabulary size between baseline and candidate**, not equality with a 200K+ production tokenizer. The earlier public o200k_base had better absolute Ukrainian token density than our v0.3 1024-token model. Never claim otherwise.
- Token count alone says nothing about fluency, reasoning accuracy, end-to-end inference speed or actual commercial billing.
- Results are corpus-specific; unrelated non-parallel UK/EN documents cannot be used to claim translation-equivalent language parity.
- Never publish the source texts, training files or learned model files from restricted corpora. Only aggregate measurements and input hashes are committed.
- Multiple tests on the same held-out set create selection risks. This follow-up is **exploratory** and must be repeated against a genuinely unseen genre-diverse corpus before any universal conclusion.

## Next milestone

Train one or more open-weight language models with adapted embeddings using a **commercially reusable, appropriately licensed** Ukrainian-heavy corpus. Measure Ukrainian perplexity, downstream benchmarks, latency, memory and total inference cost—not only token count.
