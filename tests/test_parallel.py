import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parallel_gap import ratio_ci, evaluate
import pytest


def test_parallel_ratio_and_lossless():
    enc=lambda x:[ord(ch) for ch in x]
    dec=lambda a:''.join(chr(c) for c in a)
    tokens={'unicode':(enc,dec,1114112,'test')}
    report=evaluate(['і','ії','ґ'],['a','ab','c'],tokens,repeats=200)
    assert report['unicode']['uk_en_token_ratio']==1.0
    assert report['unicode']['roundtrip_errors']==0


def test_parallel_groups_ci():
    assert ratio_ci([20,20,20],[10,10,10],['a','b','c'],100,42)==(2,2)
    with pytest.raises(ValueError):
        ratio_ci([1,2],[1],None)
    with pytest.raises(ValueError):
        ratio_ci([1,2],[1,1],['same','same'])


def test_aligned_input_does_not_silently_drop_blank_rows(tmp_path):
    from parallel_gap import read_aligned
    p=tmp_path/'aligned.txt';p.write_text('один\n\nтри\n',encoding='utf-8')
    with pytest.raises(ValueError):
        read_aligned(p)
