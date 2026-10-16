#!/usr/bin/env bash
# One-time non-commercial research workflow. Requires network for download.
# Run from repository root, after reading dataset licenses and installing dependencies.
set -euo pipefail
python corpora.py --out data/ud_raw --accept-license
python prepare_ud.py --source data/ud_raw --out data/ud_strict
python mixture.py \
  --uk data/ud_strict/uk_train.txt --en data/ud_strict/en_train.txt \
  --uk-test data/ud_strict/uk_dev.txt --en-test data/ud_strict/en_dev.txt \
  --vocab 1024 --train-budget-bytes 1500000 \
  --ratios 25 50 75 100 --out runs/ud_development
python select_ratio.py --results runs/ud_development/results.json \
  --baseline 50 --max-en-regression-pct 10 \
  --out runs/ud_development/decision.json
python final_evaluation.py \
  --decision runs/ud_development/decision.json \
  --data data/ud_strict --out runs/ud_final
