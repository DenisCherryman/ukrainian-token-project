import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utp import split_by_hash, read_lines, line_bytes, select_to_budget, WORD_RE


def test_determinism():
    data = [f'Речення номер {i}. У ньому є слова.' for i in range(100)]
    a, b = split_by_hash(data, 20, 42)
    c, d = split_by_hash(list(reversed(data)), 20, 42)
    assert set(a) == set(c)
    assert set(b) == set(d)
    assert not set(a).intersection(b)


def test_budget():
    lines = [f'Тут різні українські слова {i}' for i in range(50)]
    selected = select_to_budget(lines, 500, 42)
    assert line_bytes(selected) <= 500
    assert selected


def test_words():
    assert len(WORD_RE.findall("Українська мова — це наша спільна справа.")) == 6
    assert len(WORD_RE.findall("п'ять об’єктів")) == 2
