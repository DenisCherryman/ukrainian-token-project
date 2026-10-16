#!/usr/bin/env python3
"""Token burden on aligned Ukrainian/English translations, same tokenizer.

Do not use unrelated corpora here: the input must be exact 1:1 translations,
one text sample per line, with preserved alignment and documented license.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
from collections import defaultdict
from benchmark import make_tokenizers, sha256


def ratio_ci(uk: list[int], en: list[int], groups: list[str] | None=None,
             repeats: int=1000, seed: int=42) -> tuple[float,float]:
    if len(uk) != len(en) or not uk or sum(en) == 0:
        raise ValueError('Aligned non-empty pairs required')
    if repeats < 100:
        raise ValueError('repeats>=100')
    if groups is None:
        elems=list(zip(uk,en))
    else:
        if len(groups)!=len(uk):
            raise ValueError('Groups must align with input')
        pooled=defaultdict(lambda:[0,0])
        for a,b,g in zip(uk,en,groups):
            if not g.strip():
                raise ValueError('Empty group ID')
            pooled[g][0]+=a
            pooled[g][1]+=b
        elems=list(pooled.values())
        if len(elems)<2:
            raise ValueError('Document bootstrap requires at least 2 groups')
    rng=random.Random(seed)
    ratios=[]
    for _ in range(repeats):
        su=se=0
        for _ in range(len(elems)):
            a,b=elems[rng.randrange(len(elems))]
            su+=a;se+=b
        if se:ratios.append(su/se)
    ratios.sort()
    return (round(ratios[int((len(ratios)-1)*.025)],4),
            round(ratios[int((len(ratios)-1)*.975)],4))


def evaluate(uk_text: list[str], en_text: list[str], tokenizers: dict,
             groups=None, repeats=1000, seed=42) -> dict:
    if len(uk_text)!=len(en_text) or not uk_text:
        raise ValueError('Aligned texts must contain equal nonzero line counts')
    out={}
    for name,(enc,dec,vocab,source) in tokenizers.items():
        uk=[enc(x) for x in uk_text]
        en=[enc(x) for x in en_text]
        uk_lengths=[len(x) for x in uk]
        en_lengths=[len(x) for x in en]
        if sum(en_lengths)==0:
            raise ValueError('No English tokens')
        out[name]={
            'pairs':len(uk_text),
            'uk_tokens':sum(uk_lengths), 'en_tokens':sum(en_lengths),
            'uk_en_token_ratio':round(sum(uk_lengths)/sum(en_lengths),4),
            'ratio_95pct_ci':ratio_ci(uk_lengths,en_lengths,groups,repeats,seed),
            'bootstrap_unit':'document' if groups is not None else 'translation_pair',
            'roundtrip_errors':sum(dec(ids)!=raw for ids,raw in list(zip(uk,uk_text))+list(zip(en,en_text))),
            'vocab_size':vocab,
            'source':source,
        }
    return out


def read_aligned(path: Path) -> list[str]:
    raw = path.read_text(encoding='utf-8-sig').splitlines()
    if not raw or any(not s.strip() for s in raw):
        raise ValueError(f'{path}: aligned corpus must be nonempty and contain no blank rows')
    return raw


def main() -> None:
    p=argparse.ArgumentParser(description='Aligned UA/EN token burden')
    p.add_argument('--uk',required=True,type=Path)
    p.add_argument('--en',required=True,type=Path)
    p.add_argument('--sp',action='append',default=[])
    p.add_argument('--tiktoken',nargs='+',default=[])
    p.add_argument('--groups',type=Path,help='One document ID per aligned translation pair')
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',type=Path,default=Path('runs/parallel'))
    args=p.parse_args()
    uk=read_aligned(args.uk);en=read_aligned(args.en)
    if args.groups:
        groups=args.groups.read_text(encoding='utf-8-sig').splitlines()
    else:
        groups=None
    tok=make_tokenizers(args.sp,args.tiktoken)
    info={
        'note':'Aligned translation pairs only; ratio is NOT proof of equal semantic cost or LLM quality.',
        'uk_sha256':sha256(args.uk), 'en_sha256':sha256(args.en),
        'groups_sha256':sha256(args.groups) if args.groups else None,
        'results': evaluate(uk,en,tok,groups,args.bootstrap,args.seed)
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=['# Токенізація паралельних перекладів','','Однакові змісти, той самий токенізатор; більше 1 означає, що українською потрібно більше токенів.','',
          '| Токенізатор | Кількість пар | UA токени | EN токени | UA/EN | 95% bootstrap CI |',
          '|---|---:|---:|---:|---:|---|']
    for name,r in info['results'].items():
        rows.append(f'| {name} | {r["pairs"]} | {r["uk_tokens"]} | {r["en_tokens"]} | {r["uk_en_token_ratio"]} | {r["ratio_95pct_ci"]} |')
    rows.extend(['','Висновок обмежений наявним паралельним корпусом. Порівняння не вимірює реальні витрати API.',''])
    (args.out/'report.md').write_text('\n'.join(rows),encoding='utf-8')
    print('\n'.join(rows))


if __name__=='__main__':
    main()
