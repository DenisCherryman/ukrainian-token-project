import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from select_ratio import choose
import pytest


def test_select_with_english_threshold():
    rows=[
        {'uk_share_pct':25,'uk_tokens':150,'en_tokens':80,'model':'25.model'},
        {'uk_share_pct':50,'uk_tokens':100,'en_tokens':100,'model':'50.model'},
        {'uk_share_pct':75,'uk_tokens':70,'en_tokens':109,'model':'75.model'},
        {'uk_share_pct':100,'uk_tokens':50,'en_tokens':125,'model':'100.model'},
    ]
    result=choose(rows,50,10)
    assert result['selected_share_pct']==75
    assert result['selected_uk_reduction_pct']==30
    assert result['selected_en_regression_pct']==9
    assert choose(rows,50,0)['selected_share_pct']==50


def test_selection_rejects_duplicates_and_bad_threshold():
    row={'uk_share_pct':50,'uk_tokens':100,'en_tokens':100,'model':'50.model'}
    with pytest.raises(ValueError):
        choose([row,row])
    with pytest.raises(ValueError):
        choose([row], max_en_regression_pct=-1)


def test_does_not_select_english_heavy_share_even_if_toy_better():
    rows=[
       {'uk_share_pct':25,'uk_tokens':10,'en_tokens':50,'model':'25.model'},
       {'uk_share_pct':50,'uk_tokens':100,'en_tokens':100,'model':'50.model'},
       {'uk_share_pct':75,'uk_tokens':80,'en_tokens':150,'model':'75.model'}]
    assert choose(rows,50,10)['selected_share_pct']==50
