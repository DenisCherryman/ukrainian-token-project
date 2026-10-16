import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multivocab import check_partitions, pick_baseline_or_candidate, train_models, evaluate_frozen, public_report, render_report


def make_data(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for lang in ("uk", "en"):
        for split, count in (("train", 70), ("dev", 15), ("test", 12)):
            phrases = [("Текст дослідження про Київ, свободу та ґанок; документ номер", "uk"),
                       ("Research document about cities, trees, and freedom; number", "en")]
            sentence = phrases[lang == "en"][0]
            (root / f"{lang}_{split}.txt").write_text(
                "".join(f"{sentence} {split}-{i}.\n" for i in range(count)), encoding="utf-8")
            (root / f"{lang}_{split}.groups.txt").write_text(
                "".join(f"{split}-doc-{i}\n" for i in range(count)), encoding="utf-8")
    return root


def test_selector_en_constraint():
    assert pick_baseline_or_candidate(100, 100, 80, 108)[0] == "uk75"
    assert pick_baseline_or_candidate(100, 100, 80, 111)[0] == "uk50"
    assert pick_baseline_or_candidate(100, 100, 102, 90)[0] == "uk50"


def test_validation_prevents_document_leak(tmp_path):
    data = make_data(tmp_path)
    assert check_partitions(data)["uk_train"]["sentences"] == 70
    file = data / "uk_test.groups.txt"
    lines = file.read_text(encoding="utf-8").splitlines()
    lines[0] = "train-doc-0"
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        check_partitions(data)


def test_small_vocab_two_seeds(tmp_path):
    data = make_data(tmp_path / "data")
    choices = train_models(data, tmp_path / "work", [512], [7, 11], 6000, 10)
    assert len(choices["decisions"]) == 2
    rows = evaluate_frozen(data, choices, 100)
    assert len(rows) == 2
    report = public_report(data, choices, rows)
    assert "model_paths" not in str(report)
    assert "Noncommercial" in render_report(report)


def test_parameter_validation(tmp_path):
    data = make_data(tmp_path / "data")
    with pytest.raises(ValueError):
        train_models(data, tmp_path / "work", [383], [7], 6000, 10)
    with pytest.raises(ValueError):
        train_models(data, tmp_path / "work", [512], [7, 7], 6000, 10)
