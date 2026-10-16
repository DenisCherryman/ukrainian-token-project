import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import parse_spec, read_corpus, relative_reduction, bootstrap_ci, evaluate, aggregate
from corpora import extract_text


def test_parse_and_extract(tmp_path):
    p = tmp_path/'uk.txt'
    p.write_text('Слава Україні!\nГероям слава!\n', encoding='utf-8')
    assert parse_spec(f'uk:uk:{p}') == ('uk','uk',p)
    assert len(read_corpus(p)) == 2
    assert extract_text('# sent_id = 1\n# text = Привіт, світе!\n1\tПривіт\n') == ['Привіт, світе!']


def test_paired_bootstrap_identical():
    a = [2, 6, 4, 1]
    assert relative_reduction(a, a) == 0
    assert bootstrap_ci(a, a, 100, 7) == (0.0, 0.0)


def test_paired_bootstrap_savings():
    a, b = [10]*25, [8]*25
    assert round(relative_reduction(a,b), 4) == 20
    assert tuple(round(x, 4) for x in bootstrap_ci(a,b,200,42)) == (20,20)


def test_aggregate_roundtrip():
    report = aggregate([2,3], ['Текст один.', 'Ще один!'], 0, 500)
    assert report['tokens'] == 5
    assert report['roundtrip_errors'] == 0


def test_dataset_pair(tmp_path):
    path = tmp_path/'input.txt'
    path.write_text('Перевіряємо українську.\nНезалежна оцінка.\n', encoding='utf-8')
    tok = lambda text: [ord(c) for c in text]
    detok = lambda ids: ''.join(chr(c) for c in ids)
    tokens = {'base': (tok,detok,500,'mock'), 'copy': (tok,detok,500,'mock')}
    corpora, comparisons = evaluate([('uk', 'uk', path)], tokens, ('base','copy'), 100, 42)
    assert corpora['uk']['metrics']['base']['roundtrip_errors'] == 0
    assert comparisons['uk']['token_reduction_pct'] == 0
