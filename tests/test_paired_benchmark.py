import pytest

from paired_benchmark import document_bootstrap, evaluate, read_pairs


def test_reader(tmp_path):
    path = tmp_path / "pairs.tsv"
    path.write_text("Привіт\tHello\tarticle-1\nСвіт\tWorld\tarticle-1\n",
                    encoding="utf-8")
    rows = read_pairs(path)
    assert len(rows) == 2
    assert rows[0][2] == "article-1"


def test_rejects_invalid_lines(tmp_path):
    path = tmp_path / "pairs.tsv"
    path.write_text("Missing tab\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_pairs(path)


def test_document_bootstrap_is_deterministic():
    ua, en, documents = [7, 8, 5, 9], [4, 4, 5, 5], ["a", "a", "b", "b"]
    first = document_bootstrap(ua, en, documents, repeats=250, seed=2)
    second = document_bootstrap(ua, en, documents, repeats=250, seed=2)
    assert first == second
    assert first[0] <= first[1]


def test_one_document_no_interval():
    assert document_bootstrap([5, 6], [4, 4], ["same", "same"]) is None


def test_no_roundtrip_loss():
    rows = [("Україна", "Ukraine", "a"), ("Київ", "Kyiv", "b")]
    result = evaluate(rows, list, "".join, repeats=150)
    assert result["roundtrip_errors"] == 0
    assert result["pairs"] == 2
    assert result["documents"] == 2
