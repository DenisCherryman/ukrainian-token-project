#!/usr/bin/env python3
"""Choose a Ukrainian-heavy tokenizer on DEVELOPMENT data only.

Never run this script on the final held-out test set. Fix a threshold before
looking at final results. Decision is stored as an auditable JSON artifact.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def choose(rows: list[dict], baseline: int=50, max_en_regression_pct: float=10.0) -> dict:
    if not rows or max_en_regression_pct < 0:
        raise ValueError('Rows required and threshold must be nonnegative')
    by_share = {r['uk_share_pct']:r for r in rows}
    if len(by_share) != len(rows) or baseline not in by_share:
        raise ValueError('One unique baseline share must be present')
    ref = by_share[baseline]
    if ref['en_tokens'] <= 0 or ref['uk_tokens'] <= 0:
        raise ValueError('Baseline token totals must be positive')
    eligible = [row for row in rows if row['uk_share_pct'] >= baseline and
                row['en_tokens'] <= ref['en_tokens'] * (1 + max_en_regression_pct/100)]
    candidate = min(eligible, key=lambda r: (r['uk_tokens'], r['en_tokens'], r['uk_share_pct']))
    return {
        'baseline_share_pct': baseline,
        'selected_share_pct': candidate['uk_share_pct'],
        'max_en_regression_pct': max_en_regression_pct,
        'selected_en_regression_pct': round(100 * (candidate['en_tokens']/ref['en_tokens']-1), 4),
        'selected_uk_reduction_pct': round(100 * (1-candidate['uk_tokens']/ref['uk_tokens']), 4),
        'baseline_model': ref['model'],
        'selected_model': candidate['model'],
        'decision_data': 'dev only',
        'note': 'Tokenizer-only selection; tokenizer changes cannot be applied to deployed LLMs without adaptation.'
    }


def main() -> None:
    p=argparse.ArgumentParser(description='Select from dev-sweep without consulting test')
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--baseline',type=int,default=50)
    p.add_argument('--max-en-regression-pct',type=float,default=10.0)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=json.loads(args.results.read_text(encoding='utf-8'))
    choice=choose(result['rows'],args.baseline,args.max_en_regression_pct)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(choice,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(choice, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
