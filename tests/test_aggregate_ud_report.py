"""Tests for safe aggregate-only reporting from real research runs."""
import importlib.util
from pathlib import Path
import pytest

script = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_ud_report.py"
spec = importlib.util.spec_from_file_location("aggregate_ud_report", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    corpora = {lang: {"source": "https://example.org", "license_note": "research only",
                      "files": {"train": {"upstream_sha256": "abc", "source_url": "https://example.org"}}}
               for lang in ("uk", "en")}
    source = {"corpora": corpora}
    counts = {split: {"samples": 30, "documents": 3} for split in ("train","dev","test")}
    prep = {lang: {"splits": counts, "cross_split_duplicate_strings_removed": 0,
                    "unmarked_leading_sentences_dropped": 0} for lang in ("uk", "en")}
    sweep = {"setup": {"split_mode": "external_heldout"},
             "rows": [{"uk_share_pct": 50, "uk_train_bytes": 100,
                       "en_train_bytes": 100, "uk_tokens": 100,
                       "en_tokens": 100, "roundtrip_errors": 0}]}
    decision = {"decision_data": "dev only", "baseline_share_pct": 50,
                "selected_share_pct": 50, "max_en_regression_pct": 10,
                "selected_uk_reduction_pct": 0, "selected_en_regression_pct": 0}
    final = {"selection": {"selected_share_pct": 50},
             "comparisons": {"uk_test": {"token_reduction_pct": 0,
                                         "bootstrap_95pct_ci": [0,0],
                                         "bootstrap_unit": "document"}},
             "corpora": {"uk_test": {"language": "uk", "samples": 30,
                                     "metrics": {"baseline": {"tokens": 100,
                                                              "tokens_per_word": 2,
                                                              "roundtrip_errors": 0,
                                                              "vocab_size": 1024}}}}}
    return source, prep, sweep, decision, final


def test_safe_aggregate():
    result = module.summarize(*inputs())
    assert result["heldout_comparisons"]["uk_test"]["token_reduction_pct"] == 0
    assert "baseline_model" not in result["frozen_selection"]
    assert "document" in module.markdown(result)


def test_reject_test_selected_model():
    data = list(inputs())
    data[3]["decision_data"] = "test only"
    with pytest.raises(ValueError, match="dev only"):
        module.summarize(*data)


def test_reject_mismatched_selection():
    data = list(inputs())
    data[4]["selection"]["selected_share_pct"] = 75
    with pytest.raises(ValueError, match="mismatch"):
        module.summarize(*data)


def test_reject_roundtrip_loss():
    data = list(inputs())
    data[2]["rows"][0]["roundtrip_errors"] = 1
    with pytest.raises(ValueError, match="Roundtrip"):
        module.summarize(*data)
