# Paired Ukrainian-English token benchmark

This module complements the separate SentencePiece research prototype.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_paired_benchmark.py
python paired_benchmark.py --pairs examples/paired.sample.tsv --output runs/sample.json
```

**Input:** UTF-8 TSV, one row per aligned Ukrainian-English translation, optionally followed by a document ID as a third column. Examples are only for checking that the code runs; do not treat them as representative evidence.

**Metric:** total Ukrainian tokens / total English tokens with the **same** tokenizer on aligned translations. Values above 1 indicate higher token counts for Ukrainian. Report is accompanied by an SHA-256 of the input and a 95% paired cluster bootstrap interval when at least two documents are supplied.

**Limitations:** translations may differ in length or register; a token ratio is not an API pricing benchmark or evidence that replacing a tokenizer improves an existing model. A real study requires licensed, independent, genre-diverse data and downstream validation.

**Privacy:** do not submit confidential, licensed, or personally identifying corpora to a public repository.
