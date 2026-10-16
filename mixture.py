#!/usr/bin/env python3
"""Controlled UK/EN training-mixture sweep using fixed total bytes and vocabulary."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from utp import (read_lines, split_by_hash, line_bytes, select_to_budget,
                 write_lines, train_tokenizer, summary_sentencepiece)


def get_splits(uk_path, en_path, uk_test, en_test, seed, test_percent):
    if bool(uk_test) != bool(en_test):
        raise ValueError('--uk-test та --en-test потрібні разом')
    if uk_test:
        uk_train, u_test = read_lines(uk_path), read_lines(uk_test)
        en_train, e_test = read_lines(en_path), read_lines(en_test)
        mode = 'external_heldout'
    else:
        uk_train, u_test = split_by_hash(read_lines(uk_path), test_percent, seed)
        en_train, e_test = split_by_hash(read_lines(en_path), test_percent, seed)
        mode = 'sentence_hash_pilot'
    for lang, a, b in (('uk', uk_train, u_test), ('en', en_train, e_test)):
        overlap = {x.casefold().strip() for x in a} & {x.casefold().strip() for x in b}
        if overlap:
            raise ValueError(f'{lang}: train/test мають {len(overlap)} однакових рядків')
    return uk_train, u_test, en_train, e_test, mode


def pareto(rows):
    """Models nondominated for minimizing both UA and EN token totals."""
    return [row['uk_share_pct'] for row in rows
            if not any((r['uk_tokens'] <= row['uk_tokens'] and r['en_tokens'] <= row['en_tokens'])
                       and (r['uk_tokens'] < row['uk_tokens'] or r['en_tokens'] < row['en_tokens'])
                       for r in rows)]


def main():
    p = argparse.ArgumentParser(description='UTP: частка української під час тренування')
    p.add_argument('--uk', required=True, type=Path)
    p.add_argument('--en', required=True, type=Path)
    p.add_argument('--uk-test', type=Path)
    p.add_argument('--en-test', type=Path)
    p.add_argument('--ratios', type=int, nargs='+', default=[0,25,50,75,100],
                   help='Відсоток українських UTF-8 байтів [0..100]')
    p.add_argument('--out', type=Path, default=Path('runs/mixture'))
    p.add_argument('--vocab', type=int, default=512)
    p.add_argument('--train-budget-bytes', type=int, default=10_000_000)
    p.add_argument('--test-percent', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not (5 <= args.test_percent <= 50):
        p.error('--test-percent має бути між 5 та 50')
    if args.vocab < 384:
        p.error('--vocab >= 384 для byte fallback')
    if any(r < 0 or r > 100 for r in args.ratios):
        p.error('--ratios має бути 0..100')
    ratios = sorted(set(args.ratios))
    uk_train, uk_test, en_train, en_test, mode = get_splits(
        args.uk,args.en,args.uk_test,args.en_test,args.seed,args.test_percent)
    # Use the same total budget for all mixtures, including 0% and 100% cases.
    budget = min(args.train_budget_bytes, line_bytes(uk_train), line_bytes(en_train))
    if budget < 2048:
        raise ValueError('Надто малий бюджет для порівняння сумішей')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for share in ratios:
        n_uk = budget * share // 100
        n_en = budget - n_uk
        sampled_uk = select_to_budget(uk_train, n_uk, args.seed + 7) if n_uk else []
        sampled_en = select_to_budget(en_train, n_en, args.seed + 13) if n_en else []
        actual_uk, actual_en = line_bytes(sampled_uk), line_bytes(sampled_en)
        train = sampled_uk + sampled_en
        random.Random(args.seed).shuffle(train)
        prefix = args.out / f'uk{share:03d}'
        write_lines(args.out/f'uk{share:03d}_train.txt', train)
        model = train_tokenizer(args.out/f'uk{share:03d}_train.txt', prefix, args.vocab)
        ukm = summary_sentencepiece(model, uk_test)
        enm = summary_sentencepiece(model, en_test)
        if ukm['vocab_size'] != args.vocab or enm['vocab_size'] != args.vocab:
            raise ValueError(f'Недостатньо словника для {share}%: отримано {ukm["vocab_size"]}/{args.vocab}')
        rows.append({
            'uk_share_pct': share, 'model': str(model), 'uk_train_bytes': actual_uk,
            'en_train_bytes': actual_en, 'train_bytes': actual_uk+actual_en,
            'uk_tokens': ukm['tokens'], 'en_tokens': enm['tokens'],
            'uk_tokens_per_word': ukm['tokens_per_word'],
            'en_tokens_per_word': enm['tokens_per_word'],
            'roundtrip_errors': ukm['round_trip_errors']+enm['round_trip_errors'],
        })
    result = {
        'note': ('Descriptive tokenizer-only sweep. Sentence-split toy pilot is not representative.'
                 if mode == 'sentence_hash_pilot' else
                 'Tokenizer-only sweep on external held-out data. Ensure they are DEVELOPMENT, not final TEST.'),
        'setup': {'seed': args.seed,'vocab': args.vocab,'byte_budget_max':budget,
                  'split_mode':mode,'ratios':ratios,'uk_test_samples':len(uk_test),
                  'en_test_samples':len(en_test)},
        'pareto_uk_shares_pct':pareto(rows), 'rows':rows,
    }
    (args.out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md = ['# UTP — баланс часток української та англійської', '',
          ('**Увага:** synthetic sample; results cannot be generalized to LLMs.'
           if mode == 'sentence_hash_pilot' else
           '**Увага:** external evaluation data must be DEV, not final TEST. This is tokenizer-only.'), '',
          f'Фіксований словник: {args.vocab}; максимальний тренувальний бюджет: {budget} UTF-8 байтів.', '',
          '| Українська в навчанні (байти) | Фактично UA/EN, байтів | UA, токенів | EN, токенів | UA, токенів/слово | EN, токенів/слово |',
          '|---:|---|---:|---:|---:|---:|']
    for r in rows:
        md.append(f'| {r["uk_share_pct"]}% | {r["uk_train_bytes"]}/{r["en_train_bytes"]} | '
                  f'{r["uk_tokens"]} | {r["en_tokens"]} | {r["uk_tokens_per_word"]} | {r["en_tokens_per_word"]} |')
    md += ['', f'Недоміновані частки (Pareto): {", ".join(str(x)+"%" for x in result["pareto_uk_shares_pct"])}.',
           '', 'Дані нерівноцінні за темами й не є паралельним перекладом.',
           'Із-за довжин речень невикористані байти бюджету можуть дещо відрізнятися між сумішами.',
           'Для застосування в готовій моделі потрібна адаптація її ваг.', '']
    (args.out/'results.md').write_text('\n'.join(md),encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
