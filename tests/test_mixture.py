import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixture import pareto, get_splits


def test_pareto():
    rows = [dict(uk_share_pct=0, uk_tokens=20, en_tokens=9),
            dict(uk_share_pct=50, uk_tokens=15, en_tokens=10),
            dict(uk_share_pct=70, uk_tokens=17, en_tokens=12),
            dict(uk_share_pct=100, uk_tokens=10, en_tokens=20)]
    assert pareto(rows) == [0, 50, 100]


def test_external_holdout_rejects_leakage(tmp_path):
    train = tmp_path/'train.txt'
    test = tmp_path/'test.txt'
    train.write_text(''.join(f'Фраза {x}\n' for x in range(50)),encoding='utf-8')
    test.write_text('Фраза 15\n'+''.join(f'Тест {x}\n' for x in range(11)), encoding='utf-8')
    try:
        get_splits(train,train,test,test,42,20)
    except ValueError as e:
        assert 'train/test' in str(e)
    else:
        raise AssertionError('Leakage was not detected')
