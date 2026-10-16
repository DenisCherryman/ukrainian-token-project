#!/usr/bin/env python3
"""Independent Tatoeba CC0 out-of-domain evaluation. Aggregates only.

Do NOT pick models based on Tatoeba. Freeze the model selection on UD dev;
only then inspect Tatoeba, which has no document IDs and a different genre.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import random
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from multivocab import check_partitions, train_models
from utp import read_lines

SOURCES = {
    "uk": "https://downloads.tatoeba.org/exports/per_language/ukr/ukr_sentences_CC0.tsv.bz2",
    "en": "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_CC0.tsv.bz2",
}
CODES = {"uk": "ukr", "en": "eng"}


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def parse_row(line: str, lang_code: str) -> str | None:
    """Official exports have ID and text; some include language and modified date."""
    cells = line.rstrip("\r\n").split("\t")
    if not cells or not cells[0].isdigit():
        return None
    if len(cells) == 2:
        text = cells[1]
    elif len(cells) >= 3 and cells[1] == lang_code:
        text = cells[2]
    elif len(cells) == 3:
        text = cells[1]  # ID, sentence text, modified date.
    else:
        return None
    if not text.strip() or "\n" in text:
        return None
    return text


def load_cc0(blob: bytes, language: str, excluded: set[str], limit: int):
    if language not in CODES or limit < 20:
        raise ValueError("Unknown language or sample limit <20")
    raw = bz2.decompress(blob).decode("utf-8-sig")
    unique = {}
    n_parsed = n_invalid = n_leaked = n_duplicates = 0
    for line in raw.splitlines():
        text = parse_row(line, CODES[language])
        if text is None:
            n_invalid += 1
            continue
        n_parsed += 1
        key = normalize(text)
        if key in excluded:
            n_leaked += 1
        elif key in unique:
            n_duplicates += 1
        else:
            unique[key] = text
    # Deterministic hash-based selection, independent of the export row order.
    ordered = sorted(unique.values(), key=lambda s: (hashlib.sha256(normalize(s).encode("utf-8")).hexdigest(), s))
    texts = ordered[:limit]
    if len(texts) < 20:
        raise ValueError(f"Only {len(texts)} usable CC0 sentences for {language}, not enough for this smoke benchmark")
    return texts, {
        "compressed_sha256": hashlib.sha256(blob).hexdigest(),
        "parsed_rows": n_parsed, "invalid_rows": n_invalid,
        "removed_exact_ud_overlap": n_leaked, "removed_internal_duplicates": n_duplicates,
        "usable_unique": len(unique), "evaluated": len(texts),
        "selected_text_sha256": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(),
    }


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "UTP-CC0-research/0.5"})
    with urllib.request.urlopen(request, timeout=90) as response:
        contents = response.read(12_000_001)
    if len(contents) > 12_000_000:
        raise ValueError("Compressed archive exceeds 12 MB safety limit")
    return contents


def external_metrics(texts, baseline_path: str, candidate_path: str):
    import sentencepiece as spm
    baseline = spm.SentencePieceProcessor(model_file=baseline_path)
    candidate = spm.SentencePieceProcessor(model_file=candidate_path)
    if baseline.vocab_size() != candidate.vocab_size():
        raise ValueError("Unmatched vocabulary sizes")
    totals = [0, 0]
    for text in texts:
        for index, tokenizer in enumerate((baseline, candidate)):
            ids = tokenizer.encode(text)
            if tokenizer.decode(ids) != text:
                raise ValueError("Lossy tokenizer: roundtrip mismatch")
            totals[index] += len(ids)
    if totals[0] <= 0:
        raise ValueError("Zero baseline tokens")
    return {"baseline_tokens": totals[0], "selected_tokens": totals[1],
            "token_reduction_pct": round(100 * (1 - totals[1] / totals[0]), 3),
            "roundtrip_errors": 0,
            "effective_vocab": baseline.vocab_size()}


def study(data: Path, work: Path, seeds=(42, 173, 2026), limit_uk=3000, limit_en=2000):
    check_partitions(data)
    excluded = {
        lang: {normalize(s) for split in ("train", "dev", "test")
               for s in read_lines(data / f"{lang}_{split}.txt")}
        for lang in ("uk", "en")
    }
    testsets, provenance = {}, {}
    for language, limit in (("uk", limit_uk), ("en", limit_en)):
        testsets[language], provenance[language] = load_cc0(
            download(SOURCES[language]), language, excluded[language], limit
        )
        provenance[language]["source_url"] = SOURCES[language]
        provenance[language]["declared_license"] = "Tatoeba sentence CC0 1.0 export"
    # The selection uses UD's DEV split ONLY. No Tatoeba sample is read in train_models.
    decisions = train_models(data, work, [32768], list(seeds), 500000, 10)
    rows = []
    for choice in decisions["decisions"]:
        if not choice["matched_vocab"]:
            raise ValueError("Mismatched vocabulary; abort external evaluation")
        baseline = choice["model_paths"]["uk50"]
        selected = choice["model_paths"][choice["selected"]]
        rows.append({
            "seed": choice["seed"], "vocab": choice["vocab"],
            "selected_on_ud_dev": choice["selected"],
            "ud_dev_english_regression_pct": choice["dev_english_regression_pct"],
            "uk_cc0": external_metrics(testsets["uk"], baseline, selected),
            "en_cc0": external_metrics(testsets["en"], baseline, selected),
        })
    return {"experiment": "out_of_domain_cc0_holdout_v05",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_selection_source": "UD dev only; CC0 kept as independent test",
            "training": "UD Ukrainian-IU / English-EWT (noncommercial research), 500K raw UTF-8 byte mixture, 32K vocab",
            "sources": provenance, "rows": rows,
            "descriptive_seed_means": {
                lang: round(statistics.mean(row[f"{lang}_cc0"]["token_reduction_pct"] for row in rows), 3)
                for lang in ("uk", "en")
            },
            "limitations": [
                "Independent new SOURCE, not independent linguistically diverse documents: sentence authorship and near-duplicate sources are unknown.",
                "Exact UD sentence overlap excluded; paraphrase and training-history leakage cannot be ruled out.",
                "Only explicitly CC0-labeled Tatoeba sentences evaluated. The training data are noncommercial UD and learned weights are not redistributed.",
                "Evaluation sentence counts may be small. These seed runs reuse the same independent CC0 test and are not independent corpus replicates.",
                "No confidence intervals: individual sentences may be clustered by author or source and independence cannot be established.",
                "No GPT-6 token savings, model quality, or API pricing conclusion can be drawn.",
            ]}


def render(report):
    lines = ["# UTP v0.5 — independent Tatoeba CC0 test", "",
             "**Out-of-domain tokenization test, NOT deployed-model savings.**", "",
             "| Language | CC0 sentences | Exact UD overlaps removed |",
             "|---|---:|---:|"]
    for lang, info in report["sources"].items():
        lines.append(f"| {lang} | {info['evaluated']} | {info['removed_exact_ud_overlap']} |")
    lines += ["", "| Seed | 32K selected by UD DEV | UA CC0 token reduction | EN CC0 token reduction |",
              "|---:|---|---:|---:|"]
    for row in report["rows"]:
        lines.append(f"| {row['seed']} | {row['selected_on_ud_dev']} | "
                     f"{row['uk_cc0']['token_reduction_pct']:+.3f}% | "
                     f"{row['en_cc0']['token_reduction_pct']:+.3f}% |")
    means = report["descriptive_seed_means"]
    lines += ["", f"Descriptive mean across subset seeds: UA {means['uk']:+.3f}%, EN {means['en']:+.3f}%.",
              "", "## Cautions", ""] + ["- " + limitation for limitation in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path("runs/cc0"))
    p.add_argument("--out", type=Path, default=Path("results/cc0_external_v05.json"))
    p.add_argument("--limit-uk", type=int, default=3000)
    p.add_argument("--limit-en", type=int, default=2000)
    args = p.parse_args()
    result = study(args.data, args.work, limit_uk=args.limit_uk, limit_en=args.limit_en)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
