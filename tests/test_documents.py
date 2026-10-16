import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from prepare_ud import parse_conllu_documents, split_documents, choose_split, prepare
from benchmark import group_bootstrap_ci, read_groups, evaluate


def test_document_parser_and_repeated_id():
    text = '# newdoc id = a\n# sent_id = 1\n# text = Кіт читає.\n1\tКіт\n# newdoc id = b\n# text = Пес грає.\n'
    assert parse_conllu_documents(text,'fixture') == {'a':['Кіт читає.'], 'b':['Пес грає.']}
    with pytest.raises(ValueError):
        parse_conllu_documents('# text = Незгрупований текст.','bad')


def test_split_prevents_cross_document_leak_and_is_deterministic():
    docs = {str(i): [f'Речення з документа {i}.', 'ЗАГАЛЬНЕ РЕЧЕННЯ!'] for i in range(250)}
    a, stats = split_documents(docs, seed=11)
    b, _ = split_documents(dict(reversed(list(docs.items()))), seed=11)
    assert a == b
    assert stats['cross_split_duplicate_strings_removed'] == 1
    owners = {}
    for part, rows in a.items():
        for doc,s in rows:
            assert doc not in owners or owners[doc] == part
            owners[doc] = part
            assert s != 'ЗАГАЛЬНЕ РЕЧЕННЯ!'
    assert all(stats['splits'][p]['documents'] > 1 for p in ('train','dev','test'))


def test_bootstrap_group_constant_savings():
    a = [10,20,30,40]
    b = [8,16,24,32]
    groups = ['d1','d1','d2','d2']
    assert group_bootstrap_ci(a,b,groups,300,42) == pytest.approx((20,20))
    with pytest.raises(ValueError):
        group_bootstrap_ci(a,b,['only']*4)


def test_groups_alignment(tmp_path):
    g = tmp_path/'groups.txt'
    g.write_text('A\nB\n',encoding='utf-8')
    assert read_groups(g, 2)==['A','B']
    with pytest.raises(ValueError):
        read_groups(g,3)


def test_evaluate_cluster_mode(tmp_path):
    p=tmp_path/'uk.txt';p.write_text('Український текст.\nУкраїнський приклад.\nТретій приклад.\n',encoding='utf-8')
    g=tmp_path/'doc.txt';g.write_text('a\na\nb\n',encoding='utf-8')
    enc=lambda s:[ord(c) for c in s]
    dec=lambda ids:''.join(chr(c) for c in ids)
    tok={'same1': (enc,dec,9999,'mock'), 'same2':(enc,dec,9999,'mock')}
    _, comp=evaluate([('ua','uk',p)],tok,('same1','same2'),100,42,{'ua':g})
    assert comp['ua']['bootstrap_unit']=='document'
    assert comp['ua']['document_groups']==2


def test_preparation_from_local_files(tmp_path):
    inp=tmp_path/'raw'; inp.mkdir()
    grouped = {}
    for orig in ('train','dev','test'):
        raw = []
        for n in range({'train':0,'dev':100,'test':200}[orig],{'train':100,'dev':200,'test':300}[orig]):
            raw.append(f'# newdoc id = d{n}\n# text = Приклад документа {n}.\n')
        (inp/f'uk_iu-ud-{orig}.conllu').write_text(''.join(raw),encoding='utf-8')
    output=tmp_path/'processed'
    report=prepare(inp,output,['uk'])
    assert report['uk']['documents_input']==300
    for part in ('train','dev','test'):
        lines=(output/f'uk_{part}.txt').read_text().splitlines()
        groups=(output/f'uk_{part}.groups.txt').read_text().splitlines()
        assert len(lines)==len(groups)>0


def test_unmarked_leading_text_quarantined():
    txt='# text = Фрагмент без ID.\n# newdoc id = valid\n# text = Повний документ.\n'
    assert parse_conllu_documents(txt,'ud',allow_unmarked_leading=True)=={'valid':['Повний документ.']}
