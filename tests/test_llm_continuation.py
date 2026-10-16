import pytest

from llm_continuation import (
    CHECKPOINTS, eligible_training, render, training_schedule, verify_paired,
)


def test_bilingual_schedule_is_reproducible_and_balanced():
    uk = ["Українське " + str(i) for i in range(22)]
    en = ["English " + str(i) for i in range(22)]
    a = list(training_schedule(uk, en, 40, seed=17))
    b = list(training_schedule(uk, en, 40, seed=17))
    assert a == b
    assert sum(lang == "uk" for lang, _ in a) == 30
    assert sum(lang == "en" for lang, _ in a) == 10
    assert len({text for lang, text in a[:20] if lang == "uk"}) > 10


def test_schedule_rejects_missing_language():
    with pytest.raises(ValueError):
        list(training_schedule(["Привіт"], [], 16))
    with pytest.raises(ValueError):
        list(training_schedule([], ["Hello"], 16))


def test_training_filter_rejects_insufficient_data():
    with pytest.raises(ValueError, match="Insufficient"):
        eligible_training(["Один"], ["One"], lambda x: [1, 2, 3])


def test_training_filter_drops_overlength_examples():
    ua = ["Україна" + str(i) for i in range(25)]
    en = ["English" + str(i) for i in range(25)]
    u, e = eligible_training(ua, en, lambda x: [1] + [3]*len(x) + [2], limit=20)
    assert len(u) == 25 and len(e) == 25
    with pytest.raises(ValueError):
        eligible_training(ua, en, lambda x: [1] + [3]*len(x)*10 + [2], limit=20)


def test_paired_verification_rejects_different_counts_or_bytes():
    heldout = {"uk": ["один", "два"], "en": ["one", "two"]}
    ref = {"uk": {"texts_evaluated": 2, "bytes": 18},
           "en": {"texts_evaluated": 2, "bytes": 6}}
    result = {"0": {lang: dict(v) for lang, v in ref.items()}}
    verify_paired(ref, result, heldout)
    result["0"]["uk"]["bytes"] = 19
    with pytest.raises(ValueError, match="Unpaired byte"):
        verify_paired(ref, result, heldout)
    result["0"]["uk"]["bytes"] = 18
    result["0"]["en"]["texts_evaluated"] = 1
    with pytest.raises(ValueError, match="Unpaired sentence"):
        verify_paired(ref, result, heldout)


def test_checkpoints_fixed():
    assert CHECKPOINTS == (0, 16, 64, 128)


def test_render_no_raw_sentences():
    score = {"uk": {"bits_per_utf8_byte": 2.1, "text_tokens": 40},
             "en": {"bits_per_utf8_byte": 1.5, "text_tokens": 33}}
    report = {
        "model": "open-model", "model_revision": "abcdef",
        "original": score, "new_tokenizer_checkpoints": {"0": score, "128": score},
        "limitations": ["Not ChatGPT savings"],
    }
    out = render(report)
    assert "128 embedding steps" in out
    assert "Not ChatGPT savings" in out
