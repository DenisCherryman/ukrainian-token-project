"""UTP v0.9: strict paired document-level evaluation and bootstrap (no datasets)."""
from __future__ import annotations

import hashlib
import math
import random

SEEDS = (42, 173, 2026)
CHECKPOINTS = ("original", "0", "64", "256")
LANGUAGES = ("uk", "en")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_documents(rows: list[dict], language: str, expected: int = 30) -> None:
    """Require distinct revision-pinned pages and exact original UTF-8 text hashes."""
    if language not in LANGUAGES or len(rows) != expected:
        raise ValueError("Invalid language or holdout size")
    seen = set()
    for row in rows:
        key = (row.get("language"), row.get("page_id"))
        if key[0] != language or not isinstance(key[1], int) or key[1] <= 0 or key in seen:
            raise ValueError("Invalid or duplicated page")
        seen.add(key)
        if (row.get("source") != "Wikipedia" or row.get("license") != "CC BY-SA 4.0"
                or not isinstance(row.get("revision_id"), int) or row["revision_id"] <= 0
                or not isinstance(row.get("utf8_bytes"), int) or row["utf8_bytes"] <= 0
                or not isinstance(row.get("sentence_sha256"), str)
                or len(row["sentence_sha256"]) != 64
                or not row.get("revision_timestamp")):
            raise ValueError("Incomplete provenance metadata")
        if not row.get("url", "").startswith("https://" + language + ".wikipedia.org/"):
            raise ValueError("Missing language-specific attribution URL")


def validate_score_rows(manifest: list[dict], original: list[dict],
                        candidate: list[dict]) -> None:
    """Require exact row alignment, rather than just matching corpus totals."""
    if not (len(manifest) == len(original) == len(candidate)):
        raise ValueError("Unpaired document count")
    for meta, before, after in zip(manifest, original, candidate):
        for score in (before, after):
            if (score.get("page_id") != meta["page_id"]
                    or score.get("revision_id") != meta["revision_id"]
                    or score.get("sentence_sha256") != meta["sentence_sha256"]
                    or score.get("utf8_bytes") != meta["utf8_bytes"]):
                raise ValueError("Mismatched document or UTF-8 text")
            if (not isinstance(score.get("tokens"), int) or score["tokens"] <= 0
                    or not isinstance(score.get("nll_bits"), (int, float))
                    or not math.isfinite(score["nll_bits"]) or score["nll_bits"] < 0):
                raise ValueError("Invalid per-document likelihood or token count")


def paired_bootstrap(manifest: list[dict], original: list[dict], candidate: list[dict],
                     repeats: int = 2000, seed: int = 2026) -> dict:
    """Cluster-resample paired articles; report corpus-aggregate NLL bits / UTF-8 byte."""
    validate_score_rows(manifest, original, candidate)
    n = len(manifest)
    if n < 10 or repeats < 100:
        raise ValueError("Insufficient documents or bootstrap repeats")
    def aggregate(ids):
        bytes_total = sum(manifest[i]["utf8_bytes"] for i in ids)
        old_bits = sum(original[i]["nll_bits"] for i in ids)
        new_bits = sum(candidate[i]["nll_bits"] for i in ids)
        old_tokens = sum(original[i]["tokens"] for i in ids)
        new_tokens = sum(candidate[i]["tokens"] for i in ids)
        return ((new_bits-old_bits)/bytes_total, new_tokens-old_tokens,
                100.0*(new_tokens/old_tokens-1.0))
    observed = aggregate(range(n))
    rng = random.Random(seed)
    draws = [aggregate(rng.choices(range(n), k=n)) for _ in range(repeats)]
    ci = []
    for k in range(3):
        ordered = sorted(v[k] for v in draws)
        ci.append([round(ordered[int(.025*repeats)], 6),
                   round(ordered[min(repeats-1, int(.975*repeats))], 6)])
    return {"documents": n, "repeats": repeats, "seed": seed,
            "delta_bits_per_original_utf8_byte": round(observed[0], 6),
            "delta_bits_per_original_utf8_byte_ci95": ci[0],
            "delta_tokens": observed[1], "delta_tokens_ci95": ci[1],
            "delta_tokens_pct": round(observed[2], 6),
            "delta_tokens_pct_ci95": ci[2]}


def english_guardrail(original_bits_per_byte: float,
                      candidate_bits_per_byte: float, max_regression_pct: float = 1.0) -> dict:
    if (not math.isfinite(original_bits_per_byte) or original_bits_per_byte <= 0
            or not math.isfinite(candidate_bits_per_byte) or candidate_bits_per_byte < 0):
        raise ValueError("Invalid likelihood metric")
    pct = 100.0*(candidate_bits_per_byte/original_bits_per_byte-1.0)
    return {"regression_pct": round(pct, 5), "threshold_pct": max_regression_pct,
            "pass": pct <= max_regression_pct}
