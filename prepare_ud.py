#!/usr/bin/env python3
"""Prepare *document-disjoint* train/dev/test corpora from locally downloaded UD CoNLL-U.

Input UD file contents are not bundled with this project. This program is only
intended for appropriately licensed non-commercial research (check each corpus).
Groups are taken from # newdoc id markers. All original upstream partitions
are pooled, then partitioned again BY DOCUMENT ID using a reproducible hash.
Repeated lines appearing across distinct documents are excluded from every
partition to avoid exact string contamination.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable


UD_NAMES = {"uk": "uk_iu", "en": "en_ewt"}
PARTITIONS = ("train", "dev", "test")


def parse_conllu_documents(text: str, source: str,
                           allow_unmarked_leading: bool=False) -> dict[str, list[str]]:
    """Return literal source sentences grouped by stable, explicit document id.

    Refuse missing IDs instead of silently permitting sentence-level leakage.
    Same doc ID in different upstream files remains one group when pooled.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    doc_id = None
    for line in text.splitlines():
        if line.startswith("# newdoc id = "):
            doc_id = line[len("# newdoc id = "):].strip()
            if not doc_id:
                raise ValueError(f"{source}: empty # newdoc id")
        elif line.startswith("# text = "):
            if not doc_id:
                if allow_unmarked_leading:
                    continue  # Continuation from a source document not marked in this file; quarantine it.
                raise ValueError(f"{source}: # text found without prior # newdoc id")
            sentence = line[len("# text = "):]
            if sentence.strip():
                groups[doc_id].append(sentence)
    if not groups:
        raise ValueError(f"{source}: no complete document groups")
    return dict(groups)


def choose_split(doc_id: str, seed: int = 42, train_pct: int = 80, dev_pct: int = 10) -> str:
    key = f"{seed}:{doc_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10000
    if value < train_pct * 100:
        return "train"
    if value < (train_pct + dev_pct) * 100:
        return "dev"
    return "test"


def normalized_sentence(text: str) -> str:
    return " ".join(text.casefold().split())


def split_documents(documents: dict[str, list[str]], *, seed: int=42,
                    train_pct: int=80, dev_pct: int=10) -> tuple[dict, dict]:
    if train_pct <= 0 or dev_pct <= 0 or train_pct + dev_pct >= 100:
        raise ValueError("Train/dev/test percentages must all be positive")
    membership: dict[str, list[tuple[str, str]]] = {s: [] for s in PARTITIONS}
    for doc_id in sorted(documents):
        part = choose_split(doc_id, seed, train_pct, dev_pct)
        membership[part].extend((doc_id, s) for s in documents[doc_id])
    if any(not membership[s] for s in PARTITIONS):
        raise ValueError("One split is empty; increase the number of documents or change seed")

    # Drop cross-split duplicate sentences from ALL partitions, not just eval.
    appearances: dict[str, set[str]] = defaultdict(set)
    for part in PARTITIONS:
        for _, sent in membership[part]:
            appearances[normalized_sentence(sent)].add(part)
    leaked = {t for t, origins in appearances.items() if len(origins) > 1}
    cleaned: dict[str, list[tuple[str, str]]] = {part: [] for part in PARTITIONS}
    intra_duplicates = 0
    for part in PARTITIONS:
        seen = set()
        for doc, text in membership[part]:
            key = normalized_sentence(text)
            if key in leaked:
                continue
            if key in seen:
                intra_duplicates += 1
                continue
            seen.add(key)
            cleaned[part].append((doc, text))
    if any(not cleaned[s] for s in PARTITIONS):
        raise ValueError("A split became empty after cross-split deduplication")
    assert not (set(k for _, x in cleaned["train"] for k in [normalized_sentence(x)]) &
                set(k for _, x in cleaned["test"] for k in [normalized_sentence(x)]))
    report = {
        "documents_input": len(documents),
        "cross_split_duplicate_strings_removed": len(leaked),
        "duplicate_rows_within_splits_removed": intra_duplicates,
        "splits": {
            part: {"samples": len(rows), "documents": len(set(d for d, _ in rows))}
            for part, rows in cleaned.items()
        },
        "caveat": "Exact sentence decontamination only; near duplicates and shared authors/topics are not detected.",
    }
    return cleaned, report


def save_split(out: Path, language: str, split: dict, report: dict) -> None:
    for part, rows in split.items():
        (out/f"{language}_{part}.txt").write_text(
            "".join(text + "\n" for _, text in rows), encoding="utf-8")
        (out/f"{language}_{part}.groups.txt").write_text(
            "".join(doc + "\n" for doc, _ in rows), encoding="utf-8")
    (out/f"{language}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def prepare(source: Path, target: Path, languages: Iterable[str], seed: int=42,
            train_pct: int=80, dev_pct: int=10) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    result = {}
    for lang in languages:
        if lang not in UD_NAMES:
            raise ValueError(f"Unknown language {lang}")
        grouped = defaultdict(list)
        sources = {}
        skipped_orphans = 0
        for split in PARTITIONS:
            path = source/f"{UD_NAMES[lang]}-ud-{split}.conllu"
            raw = path.read_bytes()
            sources[split] = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
            original = raw.decode("utf-8-sig")
            docs = parse_conllu_documents(original, path.name, allow_unmarked_leading=True)
            skipped_orphans += max(0, sum(line.startswith('# text = ') for line in original.splitlines()) -
                                   sum(len(rows) for rows in docs.values()))
            for docid, lines in docs.items():
                grouped[docid].extend(lines)
        cleaned, report = split_documents(dict(grouped), seed=seed, train_pct=train_pct, dev_pct=dev_pct)
        report.update({"language": lang, "seed": seed,
                       "unmarked_leading_sentences_dropped": skipped_orphans,
                       "train_pct": train_pct, "dev_pct": dev_pct,
                       "test_pct": 100-train_pct-dev_pct, "inputs": sources,
                       "license_note": "Verify source terms; UD_Ukrainian-IU is CC BY-NC-SA 4.0"})
        save_split(target, lang, cleaned, report)
        result[lang] = report
    (target/'PREPARATION.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Document-disjoint UD split (non-commercial research)")
    p.add_argument("--source", type=Path, required=True, help="Folder with downloaded .conllu files")
    p.add_argument("--out", type=Path, default=Path("data/ud_strict"))
    p.add_argument("--languages", nargs="+", choices=tuple(UD_NAMES), default=["uk", "en"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-pct", type=int, default=80)
    p.add_argument("--dev-pct", type=int, default=10)
    args = p.parse_args()
    info = prepare(args.source, args.out, args.languages, args.seed, args.train_pct, args.dev_pct)
    print(json.dumps({lang: rec["splits"] for lang, rec in info.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
