#!/usr/bin/env python3
"""Multi-vocabulary, multi-seed tokenizer research; aggregates only.

The same document-disjoint train/dev/test split is reused in each seed. Random
seeds vary the training-sample selection; test is never used to pick a model.
Vocabulary sizes are matched *within* a baseline/candidate comparison.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from utp import read_lines, line_bytes, select_to_budget, write_lines, train_tokenizer, summary_sentencepiece
from benchmark import evaluate, make_tokenizers, read_groups


def check_partitions(data: Path) -> dict:
    """Verify document-level disjointness and reject exact text contamination."""
    counts = {}
    for lang in ("uk", "en"):
        groups = {}
        texts = {}
        for split in ("train", "dev", "test"):
            path = data / f"{lang}_{split}.txt"
            lines = read_lines(path)
            ids = read_groups(data / f"{lang}_{split}.groups.txt", len(lines))
            groups[split] = set(ids)
            texts[split] = {" ".join(x.casefold().split()) for x in lines}
            counts[f"{lang}_{split}"] = {"sentences": len(lines), "documents": len(groups[split]),
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            if groups[a] & groups[b] or texts[a] & texts[b]:
                raise ValueError(f"Document/text leakage: {lang} {a} vs {b}")
    return counts


def pick_baseline_or_candidate(base_uk, base_en, cand_uk, cand_en, en_limit_pct=10):
    """Precommitted single-vocab decision based solely on dev token counts."""
    if min(base_uk, base_en, cand_uk, cand_en) <= 0 or en_limit_pct < 0:
        raise ValueError("Nonpositive token count or negative regression limit")
    en_regression = 100 * (cand_en / base_en - 1)
    return ("uk75" if cand_uk < base_uk and en_regression <= en_limit_pct else "uk50",
            round(en_regression, 4))


def train_models(data: Path, work: Path, vocabs, seeds, byte_budget, en_limit):
    if not vocabs or not seeds or any(v < 384 for v in vocabs) or len(set(vocabs)) != len(vocabs):
        raise ValueError("Unique vocab sizes >=384 required")
    if len(set(seeds)) != len(seeds) or byte_budget < 2048:
        raise ValueError("Unique seeds and a byte budget >=2048 required")
    uk_train = read_lines(data / "uk_train.txt")
    en_train = read_lines(data / "en_train.txt")
    uk_dev = read_lines(data / "uk_dev.txt")
    en_dev = read_lines(data / "en_dev.txt")
    budget = min(byte_budget, line_bytes(uk_train), line_bytes(en_train))
    if budget < 2048:
        raise ValueError("Insufficient training data for mixture sweep")
    decisions = []
    for vocab in vocabs:
        for seed in seeds:
            folder = work / f"v{vocab}_s{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            models = {}
            dev = {}
            observed_bytes = {}
            for share in (50, 75):
                uk_budget = budget * share // 100
                en_budget = budget - uk_budget
                uk_sample = select_to_budget(uk_train, uk_budget, seed + 7)
                en_sample = select_to_budget(en_train, en_budget, seed + 13)
                mixed = uk_sample + en_sample
                random.Random(seed).shuffle(mixed)
                trainpath = folder / f"train_{share}.txt"
                write_lines(trainpath, mixed)
                model = train_tokenizer(trainpath, folder / f"uk{share}", vocab)
                uk_stat = summary_sentencepiece(model, uk_dev)
                en_stat = summary_sentencepiece(model, en_dev)
                if uk_stat["round_trip_errors"] or en_stat["round_trip_errors"]:
                    raise ValueError(f"Lossy roundtrip in DEV: vocab={vocab} seed={seed} share={share}")
                models[f"uk{share}"] = str(model)
                dev[f"uk{share}"] = {"uk_tokens": uk_stat["tokens"], "en_tokens": en_stat["tokens"],
                                    "effective_vocab": uk_stat["vocab_size"]}
                observed_bytes[f"uk{share}"] = {"uk_bytes": line_bytes(uk_sample),
                                                "en_bytes": line_bytes(en_sample)}
            actual = [dev[x]["effective_vocab"] for x in ("uk50", "uk75")]
            comparable = actual == [vocab, vocab]
            selected, regression = pick_baseline_or_candidate(
                dev["uk50"]["uk_tokens"], dev["uk50"]["en_tokens"],
                dev["uk75"]["uk_tokens"], dev["uk75"]["en_tokens"], en_limit)
            decisions.append({"vocab": vocab, "seed": seed, "actual_vocab": actual,
                              "matched_vocab": comparable, "selected": selected,
                              "dev_english_regression_pct": regression, "dev": dev,
                              "observed_bytes": observed_bytes, "model_paths": models})
    return {"seed_type": "training-subset resampling; NOT independent documents",
            "max_training_bytes": budget, "en_limit_pct": en_limit,
            "selection_uses": "dev_only", "decisions": decisions}


def evaluate_frozen(data: Path, decision_bundle, repeats=1000):
    """One held-out evaluation *after all dev-only model choices are frozen*."""
    if decision_bundle.get("selection_uses") != "dev_only":
        raise ValueError("Selection must be frozen on dev data")
    rows = []
    for dec in decision_bundle["decisions"]:
        row = {key: dec[key] for key in ("vocab", "seed", "actual_vocab", "matched_vocab",
                                         "selected", "dev_english_regression_pct", "dev", "observed_bytes")}
        if not dec["matched_vocab"]:
            row["status"] = "skipped_unequal_effective_vocab"
            rows.append(row)
            continue
        tokenizers = make_tokenizers([f"baseline:{dec['model_paths']['uk50']}",
                                      f"candidate:{dec['model_paths'][dec['selected']}"], [])
        datasets = [(f"{lang}_test", lang, data / f"{lang}_test.txt") for lang in ("uk", "en")]
        groupfiles = {f"{lang}_test": data / f"{lang}_test.groups.txt" for lang in ("uk", "en")}
        metrics, comparisons = evaluate(datasets, tokenizers, ("baseline", "candidate"),
                                        repeats, 1729, groupfiles)
        for key in metrics:
            for stats in metrics[key]["metrics"].values():
                if stats["roundtrip_errors"]:
                    raise ValueError(f"Lossy heldout roundtrip in {key}")
        row["status"] = "evaluated"
        row["test"] = {lang: {"baseline_tokens": metrics[f"{lang}_test"]["metrics"]["baseline"]["tokens"],
                              "selected_tokens": metrics[f"{lang}_test"]["metrics"]["candidate"]["tokens"],
                              "reduction_pct": comparisons[f"{lang}_test"]["token_reduction_pct"],
                              "ci_95pct": comparisons[f"{lang}_test"]["bootstrap_95pct_ci"],
                              "document_groups": comparisons[f"{lang}_test"]["document_groups"]}
                       for lang in ("uk", "en")}
        rows.append(row)
    return rows


def public_report(data: Path, decisions, rows):
    clean_decisions = [{k: v for k, v in row.items() if k != "model_paths"}
                       for row in decisions["decisions"]]
    report = {"status": "real_corpus_multivocab_research",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection": "dev_only",
              "vocabulary_comparison": "matched per candidate-baseline pair",
              "training_limit_bytes": decisions["max_training_bytes"],
              "max_en_regression_pct_on_dev": decisions["en_limit_pct"],
              "seeds_note": decisions["seed_type"], "corpus": check_partitions(data),
              "dev_decisions": clean_decisions, "heldout": rows,
              "limitations": ["Noncommercial dataset; source text and model artifacts not released.",
                              "This is NOT ChatGPT or GPT-6 token, cost, speed or quality improvement.",
                              "Seeds resample the same documents; their outcomes are NOT independent corpus replications.",
                              "Test documents are shared across all configurations. Bootstrap CI does not account for domain shift.",
                              "If a requested vocabulary size cannot be reached by BOTH models, mark it skipped."]}
    return report


def render_report(report):
    lines = ["# UTP v0.4 — vocabulary scaling on actual UD text", "",
             "**Noncommercial tokenizer experiment; not ChatGPT savings.**", "",
             "Requested vocabularies and independently sampled training subsets share the same held-out test set.", "",
             "| Vocab requested | Seed | Actual vocab baseline/candidate | Selected on dev | UK test reduction | EN test reduction |",
             "|---:|---:|---|---|---:|---:|"]
    grouped = {}
    for row in report["heldout"]:
        if row["status"] != "evaluated":
            lines.append(f"| {row['vocab']} | {row['seed']} | {row['actual_vocab']} | SKIPPED | — | — |")
            continue
        lines.append(f"| {row['vocab']} | {row['seed']} | {row['actual_vocab']} | {row['selected']} | "
                     f"{row['test']['uk']['reduction_pct']:+.3f}% | {row['test']['en']['reduction_pct']:+.3f}% |")
        grouped.setdefault(row["vocab"], []).append(row)
    lines += ["", "## Mean reductions across subset-selection seeds (descriptive)", "",
              "| Vocab | Seeds evaluated | Ukrainian | English |", "|---:|---:|---:|---:|"]
    for vocab, rows in sorted(grouped.items()):
        meanuk = statistics.mean(x["test"]["uk"]["reduction_pct"] for x in rows)
        meanen = statistics.mean(x["test"]["en"]["reduction_pct"] for x in rows)
        lines.append(f"| {vocab} | {len(rows)} | {meanuk:+.3f}% | {meanen:+.3f}% |")
    lines += ["", "## Limits", ""] + ["- " + x for x in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("runs/multivocab"))
    parser.add_argument("--out", type=Path, default=Path("results/multivocab_v04.json"))
    parser.add_argument("--vocabs", type=int, nargs="+", default=[1024, 4096, 8192, 16384, 32768])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 173, 2026])
    parser.add_argument("--train-budget-bytes", type=int, default=500_000)
    parser.add_argument("--max-en-regression-pct", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    check_partitions(args.data)
    choice = train_models(args.data, args.work, args.vocabs, args.seeds,
                          args.train_budget_bytes, args.max_en_regression_pct)
    # Freeze the whole DEV selection before accessing held-out token scores.
    frozen = args.work / "frozen_dev_choices.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(choice, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = evaluate_frozen(args.data, choice, args.bootstrap)
    report = public_report(args.data, choice, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render_report(report), encoding="utf-8")
    print(render_report(report))


if __name__ == "__main__":
    main()
