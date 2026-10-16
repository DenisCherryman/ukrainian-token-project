# Зміни

## v0.3 — 2026-10-17

- Підготовка train/dev/test з групуванням за ID документів і вилученням exact-дублікатів між підвибірками.
- Збереження оригінальних CoNLL-U і SHA-256, параметри вибору upstream git revision.
- Кластерний bootstrap у `benchmark.py` за опційними документними ID.
- Попередньо фіксоване правило вибору параметрів на dev (`select_ratio.py`) і фінальний заморожений test (`final_evaluation.py`).
- Додано паралельний UA/EN бенчмарк (`parallel_gap.py`).
- Реалізовано tests і локальну синтетичну end-to-end перевірку; реальні UD не запускалися в цьому середовищі.

## v0.2

- SentencePiece-BPE sweep, text-level bootstrap, tiktoken references, UD downloader, tests, CI.
