#!/usr/bin/env python3
"""Publish aggregate-only results of the non-commercial UD experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def summarize(source, prep, sweep, decision, final, references=None):
    if decision.get("decision_data") != "dev only":
        raise ValueError("Selection must happen on dev only")
    if final.get("selection", {}).get("selected_share_pct") != decision.get("selected_share_pct"):
        raise ValueError("Frozen selection mismatch")
    if sweep.get("setup", {}).get("split_mode") != "external_heldout":
        raise ValueError("Development requires an external heldout split")
    if not final.get("comparisons"):
        raise ValueError("Missing heldout comparisons")
    if set(prep) != {"uk", "en"}:
        raise ValueError("Both language reports are required")
    result = {
        "status": "measured_noncommercial_research",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            language: {
                "url": corpus["source"], "license": corpus["license_note"],
                "files": {part: {"sha256": f["upstream_sha256"], "url": f["source_url"]}
                          for part, f in corpus["files"].items()},
            } for language, corpus in source["corpora"].items()
        },
        "splits": {
            lang: {"document_disjoint": True, "counts": prep[lang]["splits"],
                   "removed_duplicate_strings": prep[lang]["cross_split_duplicate_strings_removed"],
                   "skipped_orphans": prep[lang]["unmarked_leading_sentences_dropped"]}
            for lang in ("uk", "en")
        },
        "development_setup": sweep["setup"],
        "development_sweep": [
            {k: row[k] for k in ("uk_share_pct", "uk_train_bytes", "en_train_bytes",
                                 "uk_tokens", "en_tokens", "roundtrip_errors")}
            for row in sweep["rows"]
        ],
        "frozen_selection": {
            k: decision[k] for k in ("baseline_share_pct", "selected_share_pct",
                                     "max_en_regression_pct", "selected_uk_reduction_pct",
                                     "selected_en_regression_pct")
        },
        "heldout_comparisons": final["comparisons"],
        "heldout_metrics": {
            name: {"language": corpus["language"], "samples": corpus["samples"],
                   "metrics": {label: {field: value[field] for field in (
                       "tokens", "tokens_per_word", "roundtrip_errors", "vocab_size")}
                               for label, value in corpus["metrics"].items()}}
            for name, corpus in final["corpora"].items()
        },
        "reference_encodings": references or {},
        "limitations": [
            "Tokenizer-only experiment, NOT API price/quality/latency or GPT-6 saving.",
            "UD_Ukrainian-IU is CC BY-NC-SA 4.0; do not redistribute source texts or learned model weights.",
            "Ukrainian and English datasets are not sentence-aligned translations.",
            "tiktoken vocabularies differ in size; references are not controlled comparisons.",
            "Exact sentence deduplication does not prevent paraphrase/topic leakage.",
            "Single small corpus cannot represent all Ukrainian language use.",
        ],
    }
    if any(row["roundtrip_errors"] for row in result["development_sweep"]):
        raise ValueError("Roundtrip loss in development")
    for corpus in result["heldout_metrics"].values():
        if any(m["roundtrip_errors"] for m in corpus["metrics"].values()):
            raise ValueError("Roundtrip loss in heldout test")
    return result


def markdown(result):
    lines = ["# Ukrainian Token Project — real-corpus benchmark", "",
             "**Non-commercial tokenizer-only experiment. NOT ChatGPT savings.**", "",
             "Executed: " + result["executed_at_utc"], "",
             "## Document-disjoint splits", "",
             "| Language | Train (sentences/docs) | Dev | Test |", "|---|---:|---:|---:|"]
    for lang, obj in result["splits"].items():
        counts = obj["counts"]
        values = [str(counts[s]["samples"]) + "/" + str(counts[s]["documents"]) for s in ("train", "dev", "test")]
        lines.append("| " + lang + " | " + " | ".join(values) + " |")
    sel = result["frozen_selection"]
    lines += ["", "## Frozen model selection (dev only)", "",
              f"Baseline UA share: {sel['baseline_share_pct']}%; selected: {sel['selected_share_pct']}%.",
              "", "## Held-out test results", ""]
    for name, info in result["heldout_comparisons"].items():
        lines.append(f"- **{name}: {info['token_reduction_pct']:+.3f}%** reduction; "
                     f"95% CI {info['bootstrap_95pct_ci']}, bootstrap unit: {info['bootstrap_unit']}.")
    lines += ["", "## Limitations", ""]
    lines += ["- " + entry for entry in result["limitations"]]
    lines += ["", "See matching JSON for aggregate numbers and SHA-256 source checksums.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/ud_raw/SOURCES.json"))
    p.add_argument("--preparation", type=Path, default=Path("data/ud_strict/PREPARATION.json"))
    p.add_argument("--sweep", type=Path, default=Path("runs/ud_development/results.json"))
    p.add_argument("--decision", type=Path, default=Path("runs/ud_development/decision.json"))
    p.add_argument("--final", type=Path, default=Path("runs/ud_final/report.json"))
    p.add_argument("--references", type=Path, default=Path("runs/ud_final/reference.json"))
    p.add_argument("--out", type=Path, default=Path("results/ud_iu_ewt_v03.json"))
    args = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    result = summarize(read(args.source), read(args.preparation), read(args.sweep),
                       read(args.decision), read(args.final),
                       read(args.references) if args.references.is_file() else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
