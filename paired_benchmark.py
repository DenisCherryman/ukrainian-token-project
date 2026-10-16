#!/usr/bin/env python3
"""Measure tokenizer efficiency on aligned Ukrainian-English translations.

Input: UTF-8 TSV with Ukrainian, English, and optional document ID columns.
An aligned translation corpus is required for a meaningful language comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def read_pairs(path: Path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        columns = line.split("\t")
        if len(columns) not in (2, 3) or not all(col.strip() for col in columns):
            raise ValueError(f"{path}:{number}: expected 2 or 3 nonempty TSV columns")
        rows.append((columns[0], columns[1],
                     columns[2] if len(columns) == 3 else str(number)))
    if not rows:
        raise ValueError("Expected at least one aligned translation pair")
    return rows


def document_bootstrap(uk_counts, en_counts, ids, repeats=2000, seed=42):
    """Paired cluster bootstrap; returns None if fewer than 2 document IDs."""
    if not uk_counts or not (len(uk_counts) == len(en_counts) == len(ids)):
        raise ValueError("Expected equal-length nonempty inputs")
    if not sum(en_counts):
        raise ValueError("English token count is zero")
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    documents = defaultdict(lambda: [0, 0])
    for u, e, doc_id in zip(uk_counts, en_counts, ids):
        documents[doc_id][0] += u
        documents[doc_id][1] += e
    population = list(documents.values())
    if len(population) < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        sample = [population[rng.randrange(len(population))] for _ in population]
        denominator = sum(x[1] for x in sample)
        if denominator > 0:
            draws.append(sum(x[0] for x in sample) / denominator)
    if not draws:
        return None
    draws.sort()
    return [round(draws[int(0.025 * (len(draws) - 1))], 4),
            round(draws[int(0.975 * (len(draws) - 1))], 4)]


def evaluate(rows, encode, decode, repeats=2000, seed=42):
    uk_encoded = [encode(row[0]) for row in rows]
    en_encoded = [encode(row[1]) for row in rows]
    uk_counts = [len(item) for item in uk_encoded]
    en_counts = [len(item) for item in en_encoded]
    if not sum(en_counts):
        raise ValueError("English token count is zero")
    errors = sum(int(decode(uk) != row[0]) + int(decode(en) != row[1])
                 for row, uk, en in zip(rows, uk_encoded, en_encoded))
    return {
        "pairs": len(rows),
        "documents": len(set(row[2] for row in rows)),
        "uk_tokens": sum(uk_counts),
        "en_tokens": sum(en_counts),
        "uk_to_en_ratio": round(sum(uk_counts) / sum(en_counts), 4),
        "ratio_95pct_ci": document_bootstrap(
            uk_counts, en_counts, [row[2] for row in rows], repeats, seed
        ),
        "roundtrip_errors": errors,
    }


def run(path, encodings, output, repeats=2000, seed=42):
    import tiktoken
    rows = read_pairs(path)
    report = {
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "method": "aligned UA/EN translations, same tokenizer, document bootstrap",
        "caution": "Token ratios alone do not imply lower LLM prices, better accuracy or faster inference.",
        "results": {},
    }
    for name in encodings:
        tokenizer = tiktoken.get_encoding(name)
        report["results"][name] = evaluate(
            rows, tokenizer.encode, tokenizer.decode, repeats, seed
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--encodings", nargs="+",
                        default=["cl100k_base", "o200k_base"])
    parser.add_argument("--output", type=Path, default=Path("runs/paired.json"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.pairs, args.encodings, args.output,
                         args.bootstrap, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
