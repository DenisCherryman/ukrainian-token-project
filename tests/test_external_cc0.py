import bz2
import json
import pytest

from external_cc0 import parse_row, load_cc0, external_metrics, normalize, render


def test_parser_accepts_tatoeba_formats():
    assert parse_row("7\tukr\tПривіт!\t2026-10-01", "ukr") == "Привіт!"
    assert parse_row("7\tПривіт!", "ukr") == "Привіт!"
    assert parse_row("7\tukr\tПривіт!", "ukr") == "Привіт!"
    assert parse_row("7\tПривіт!\t2026-10-01", "ukr") == "Привіт!"
    assert parse_row("bad\tПривіт!", "ukr") is None


def test_cc0_overlap_and_dedup():
    lines = []
    for n in range(30):
        lines.append(f"{n+1}\tukr\tЦе український приклад номер {n}.\t2026-10-01")
    lines.append("101\tukr\tце український приклад номер 2.\t2026-10-01")
    blob = bz2.compress(("\n".join(lines) + "\n").encode("utf-8"))
    texts, stats = load_cc0(blob, "uk", {normalize("Це український приклад номер 1.")}, 25)
    assert len(texts) == 25
    assert stats["removed_exact_ud_overlap"] == 1
    assert stats["removed_internal_duplicates"] == 1
    assert stats["evaluated"] == 25
    assert "source_url" not in stats


def test_cc0_limit_guard():
    with pytest.raises(ValueError, match="sample limit"):
        load_cc0(bz2.compress(b""), "uk", set(), 1)


def test_cc0_reject_short_corpus():
    blob = bz2.compress("1\tukr\tОдне речення.\n".encode("utf-8"))
    with pytest.raises(ValueError, match="not enough"):
        load_cc0(blob, "uk", set(), 500)


def test_render_report_no_raw_text():
    result = {
        "sources": {"uk": {"evaluated": 30, "removed_exact_ud_overlap": 2},
                    "en": {"evaluated": 100, "removed_exact_ud_overlap": 0}},
        "rows": [{"seed": 42, "selected_on_ud_dev": "uk75",
                  "uk_cc0": {"token_reduction_pct": 4.1},
                  "en_cc0": {"token_reduction_pct": -5.0}}],
        "descriptive_seed_means": {"uk": 4.1, "en": -5.0},
        "limitations": ["Independent-source descriptive measurement, not GPT savings"],
    }
    text = render(result)
    assert "+4.100%" in text and "30" in text
    assert "GPT savings" in text
