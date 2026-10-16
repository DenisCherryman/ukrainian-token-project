#!/usr/bin/env python3
"""Evaluate a preselected tokenizer on an untouched document-disjoint TEST set.

Requires a frozen selection from select_ratio.py; does not choose a model using
any test-set information. Caller is responsible for ensuring test not used prior.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from benchmark import evaluate, render_md, make_tokenizers, sha256


def main() -> None:
    p=argparse.ArgumentParser(description='Frozen-selection held-out UD evaluation')
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/final_evaluation'))
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.out.exists() and list(args.out.glob('report*')):
        p.error('Final test result already exists; use a new --out and document the rerun reason.')
    decision=json.loads(args.decision.read_text(encoding='utf-8'))
    if decision.get('decision_data')!='dev only':
        p.error('Selection must be frozen from development data only')
    tokenizer_specs=[f"baseline:{decision['baseline_model']}",f"selected:{decision['selected_model']}"]
    tok=make_tokenizers(tokenizer_specs,[])
    datasets=[]
    groups={}
    for language in ('uk','en'):
        dataset_name=f'{language}_test'
        f=args.data/f'{language}_test.txt'
        g=args.data/f'{language}_test.groups.txt'
        if not f.is_file() or not g.is_file():
            p.error(f'Missing test file or group IDs for {language}')
        datasets.append((dataset_name,language,f))
        groups[dataset_name]=g
    metrics,comparisons=evaluate(datasets,tok,('baseline','selected'),args.bootstrap,args.seed,groups)
    report={
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'selection':decision,
        'selection_file_sha256':sha256(args.decision),
        'interpretation':'Document-cluster bootstrap; tokenizer-only, no LLM deployment or price claims.',
        'corpora':metrics, 'comparisons':comparisons,
        'models':{name:{'vocab_size':spec[2],'source':spec[3]} for name,spec in tok.items()}
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.out/'report.md').write_text(render_md(report)+'\n',encoding='utf-8')
    print(render_md(report))


if __name__=='__main__':
    main()
