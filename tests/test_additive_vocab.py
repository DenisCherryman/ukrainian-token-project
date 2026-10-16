"""Fast regression tests: candidate selection and old English IDs remain intact."""
import pytest
from additive_vocab import (choose_ukrainian_pieces, add_pieces,
                            verify_old_english_ids, assert_paired, markdown,
                            STEPS)


class FakeTokenizer:
    def __init__(self):
        self.vocab = ["<unk>", "the", "is", "місто", "Київ"]
        self.added = {}
    def __len__(self):
        return len(self.vocab) + len(self.added)
    def encode(self, text, add_special_tokens=False):
        # No accidental Cyrillic segmentation inside English.
        return [1] if text.isascii() else ([4, 3] if text == "Україна" else [3, 4])
    def convert_ids_to_tokens(self, idx):
        return (self.vocab + list(self.added))[idx]
    def convert_tokens_to_ids(self, token):
        return self.vocab.index(token) if token in self.vocab else self.added[token]
    def add_tokens(self, new):
        for p in new:
            name = str(p)
            self.added[name] = len(self.vocab) + len(self.added)
        return len(new)


def test_select_cyrillic_parts_reproducibly():
    t = FakeTokenizer()
    corpus = ["Українська мова, українська мова!"]*6 + ["Захистимо українську культуру."]*4
    a = choose_ukrainian_pieces(corpus, t, count=15)
    b = choose_ukrainian_pieces(list(reversed(corpus)), t, count=15)
    assert a == b and len(a) == 15
    assert all("a" not in s for s in a)


def test_additions_keep_unchanged_original_ids():
    t=FakeTokenizer()
    old=add_pieces(t, ["Українська", "привіт"])
    assert old == 5 and len(t) == 7
    assert t.convert_tokens_to_ids("Українська") == 5
    with pytest.raises(ValueError):
        add_pieces(t, ["hello"])


def test_invariant_checks_old_english_token_ids():
    t=FakeTokenizer()
    a=FakeTokenizer()
    add_pieces(a, ["Українська"])
    assert verify_old_english_ids(t, a, ["Hello there."]*33) == 33
    with pytest.raises(ValueError):
        verify_old_english_ids(t, a, ["English"]*10)


def test_strict_paired_denominators():
    wanted={"uk":{"count":2,"bytes":50},"en":{"count":3,"bytes":15}}
    results={"baseline":{"uk":{"texts_evaluated":2,"bytes":50},
                          "en":{"texts_evaluated":3,"bytes":15}}}
    assert_paired(results,wanted)
    results["baseline"]["en"]["bytes"]=17
    with pytest.raises(ValueError,match="Unpaired"):
        assert_paired(results,wanted)


def test_checkpoints_and_no_claims():
    assert STEPS == (0,64,256)
    state={"uk":{"text_tokens":10,"bits_per_utf8_byte":2.4},
           "en":{"text_tokens":8,"bits_per_utf8_byte":1.3}}
    fake={"model_id":"test","pinned_revision":"abc","token_additions":{"old_vocab":10,"added":1,"new_vocab":11},
          "english_token_id_invariance_checked":30,"original_model":state,
          "augmented_model":{"0":state,"256":state},"limitations":["No GPT-6 billing savings"]}
    result=markdown(fake)
    assert "No GPT-6 billing savings" in result and "256" in result
