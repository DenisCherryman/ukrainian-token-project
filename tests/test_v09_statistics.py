"""Regression tests for v0.9 paired evaluation helpers."""
import pytest
from v09_protocol import sha256_text, english_guardrail, validate_score_rows, paired_bootstrap


def _data():
    manifest = [
        {"page_id": i + 1, "revision_id": 100 + i,
         "sentence_sha256": sha256_text(str(i)), "utf8_bytes": 100 + i}
        for i in range(12)
    ]
    before = [dict(m, tokens=50, nll_bits=m["utf8_bytes"] * 2.0) for m in manifest]
    after = [dict(m, tokens=45, nll_bits=m["utf8_bytes"] * 1.8) for m in manifest]
    return manifest, before, after


def test_paired_bootstrap():
    m, a, b = _data()
    result = paired_bootstrap(m, a, b, repeats=200)
    assert result["delta_bits_per_original_utf8_byte"] == -0.2
    assert result["delta_tokens"] == -60
    assert result == paired_bootstrap(m, a, b, repeats=200)


def test_mismatched_revision_rejected():
    m, a, b = _data()
    b[0]["revision_id"] += 1
    with pytest.raises(ValueError):
        validate_score_rows(m, a, b)


def test_english_regression():
    assert english_guardrail(1.0, 1.005)["pass"]
    assert not english_guardrail(1.0, 1.02)["pass"]


def test_nonfinite_score_rejected():
    m, a, b = _data()
    b[1]["nll_bits"] = float("nan")
    with pytest.raises(ValueError):
        validate_score_rows(m, a, b)
