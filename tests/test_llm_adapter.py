"""Fast invariants for v0.6 CPU LLM adapter (no model download needed)."""
import pytest
from llm_adapter import byte_budget_split, disjoint_cc0, markdown


def test_sha_split_deterministic_and_disjoint():
    corpus = [f"Речення номер {i}. Український приклад." for i in range(250)]
    first = disjoint_cc0(corpus)
    second = disjoint_cc0(list(reversed(corpus)))
    assert set(first[0]) == set(second[0])
    assert set(first[1]) == set(second[1])
    assert not (set(first[0]) & set(first[1]))


def test_training_mix_respects_byte_budget():
    ukrainian = [f"Українське речення про Мову № {i}." for i in range(100)]
    english = [f"An English sentence about language number {i}." for i in range(100)]
    ua, en = byte_budget_split(ukrainian, english, 8000, share=75)
    assert ua and en
    assert len("".join(x+"\n" for x in ua).encode("utf-8")) <= 6000
    assert len("".join(x+"\n" for x in en).encode("utf-8")) <= 2000


def test_bad_train_parameters():
    with pytest.raises(ValueError):
        byte_budget_split(["one"], ["two"], 100)
    with pytest.raises(ValueError):
        disjoint_cc0(["one", "two"], train_pct=99)


def test_markdown_has_distinct_quality_metric():
    report = {"pretrained_model": "small-open-weight", "model_revision": "sha",
              "tokenizer_training": {"vocab": 1024},
              "adapter_training_steps": 2, "results": {
                  kind: {"uk": {"bits_per_utf8_byte": 2.7},
                         "en": {"bits_per_utf8_byte": 2.2}}
                  for kind in ("original_tokenizer", "new_vocab_unadapted", "new_vocab_after_training")},
              "limitations": ["Do not claim ChatGPT savings"]}
    result = markdown(report)
    assert "bits/UTF-8 byte" in result
    assert "Do not claim ChatGPT savings" in result
    assert "new_vocab_after_training" in result
