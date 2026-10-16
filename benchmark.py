#!/usr/bin/env python3
"""UTP tokenizer benchmark: corpus-oriented, paired bootstrap, lossless round-trips.

This is tokenization-only evaluation; it does not measure LLM quality, latency,
API billing, or compatibility with a deployed model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

WORD_RE = re.compile(r"\w+(?:[’'\-]\w+)*", re.UNICODE)


def parse_spec(spec: str) -> tuple[str, str, Path]:
    try:
        name, language, filename = spec.split(':', 2)
    except ValueError as exc:
        raise ValueError('Корпус: NAME:uk|en:PATH') from exc
    if not name or language not in {'uk', 'en'} or not filename:
        raise ValueError('Корпус: NAME:uk|en:PATH')
    return name, language, Path(filename)


def parse_model(spec: str) -> tuple[str, Path]:
    name, _, path = spec.partition(':')
    if not name or not path:
        raise ValueError('Модель: NAME:PATH')
    return name, Path(path)


def read_corpus(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not lines:
        raise ValueError(f'Порожній корпус: {path}')
    return lines


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_tokenizers(sp_specs, tk_names):
    tokenizers = {}
    try:
        import sentencepiece as spm
    except ImportError:
        if sp_specs:
            raise RuntimeError('pip install sentencepiece')
    else:
        for spec in sp_specs:
            name, model_path = parse_model(spec)
            if name in tokenizers:
                raise ValueError(f'Неунікальна назва токенізатора: {name}')
            processor = spm.SentencePieceProcessor(model_file=str(model_path))
            tokenizers[name] = (processor.encode, processor.decode, processor.vocab_size(), str(model_path))
    if tk_names:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError('Для --tiktoken встановіть pip install tiktoken') from exc
        for name in tk_names:
            label = f'tiktoken:{name}'
            enc = tiktoken.get_encoding(name)
            tokenizers[label] = (lambda s, e=enc: e.encode(s, disallowed_special=()), enc.decode, enc.n_vocab, name)
    if not tokenizers:
        raise ValueError('Задайте --sp або --tiktoken')
    return tokenizers


def aggregate(lengths: list[int], lines: list[str], errors: int, vocab_size: int) -> dict:
    ntokens = sum(lengths)
    words = sum(len(WORD_RE.findall(line)) for line in lines)
    utf8_bytes = sum(len(line.encode('utf-8')) for line in lines)
    chars = sum(len(line) for line in lines)
    return {
        'samples': len(lines), 'tokens': ntokens, 'words': words,
        'utf8_bytes': utf8_bytes, 'unicode_chars': chars,
        'tokens_per_word': round(ntokens/words, 5) if words else None,
        'bytes_per_token': round(utf8_bytes/ntokens, 5) if ntokens else None,
        'chars_per_token': round(chars/ntokens, 5) if ntokens else None,
        'roundtrip_errors': errors, 'vocab_size': vocab_size,
    }


def relative_reduction(baseline: list[int], candidate: list[int]) -> float:
    a, b = sum(baseline), sum(candidate)
    if a <= 0:
        raise ValueError('Порожня базова токенізація')
    return 100.0*(1.0-b/a)


def bootstrap_ci(baseline: list[int], candidate: list[int], repeats: int, seed: int) -> tuple[float, float]:
    """Paired resampling of rows; valid only when row-level samples are independent."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError('Масиви повинні бути непорожніми і рівної довжини')
    if repeats < 100:
        raise ValueError('bootstrap repeats >= 100')
    rng = random.Random(seed)
    n = len(baseline)
    observations = []
    for _ in range(repeats):
        ba, ca = 0, 0
        for _ in range(n):
            index = rng.randrange(n)
            ba += baseline[index]
            ca += candidate[index]
        if ba:
            observations.append(100*(1-ca/ba))
    observations.sort()
    return observations[int((len(observations)-1)*.025)], observations[int((len(observations)-1)*.975)]


def group_bootstrap_ci(baseline: list[int], candidate: list[int],
                       group_ids: list[str], repeats: int=1000,
                       seed: int=42) -> tuple[float,float]:
    """Paired cluster bootstrap, drawing documents with replacement.

    Each document contributes its full token counts per resample.
    This is superior to row bootstrap when rows of the same document
    are dependent, but cannot correct underlying corpus selection bias.
    """
    if len(baseline) != len(candidate) or len(baseline) != len(group_ids):
        raise ValueError('token lengths and group IDs must align')
    if repeats < 100:
        raise ValueError('bootstrap repeats >=100')
    sums = defaultdict(lambda: [0,0])
    for a,b,g in zip(baseline,candidate,group_ids):
        if not g.strip():
            raise ValueError('empty document ID')
        sums[g][0] += a
        sums[g][1] += b
    if len(sums) < 2:
        raise ValueError('cluster bootstrap needs at least 2 documents')
    docs = sorted(sums)
    n = len(docs)
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        base = cand = 0
        for _ in range(n):
            a,b = sums[docs[rng.randrange(n)]]
            base += a
            cand += b
        if base:
            draws.append(100.0 * (1.0-cand/base))
    draws.sort()
    return draws[int((len(draws)-1)*.025)], draws[int((len(draws)-1)*.975)]


def read_groups(path: Path, count: int) -> list[str]:
    groups = path.read_text(encoding='utf-8-sig').splitlines()
    if len(groups) != count:
        raise ValueError(f'{path}: groups count {len(groups)} != sample count {count}')
    if any(not g.strip() for g in groups):
        raise ValueError(f'{path}: contains empty group IDs')
    return groups

def evaluate(datasets, tokenizers, compare=None, repeats=1000, seed=42, groups=None):
    metrics = {}
    comparisons = {}
    groups = groups or {}
    for name, language, path in datasets:
        lines = read_corpus(path)
        token_lengths = {}
        metrics[name] = {'language': language, 'file_sha256': sha256(path), 'metrics': {},
                         'unique_samples': len(set(lines)), 'samples': len(lines)}
        for tokenizer, (encode, decode, vocab, _) in tokenizers.items():
            lengths = []
            errors = 0
            for line in lines:
                ids = encode(line)
                lengths.append(len(ids))
                if decode(ids) != line:
                    errors += 1
            token_lengths[tokenizer] = lengths
            metrics[name]['metrics'][tokenizer] = aggregate(lengths, lines, errors, vocab)
        if compare:
            base, candidate = compare
            if base not in token_lengths or candidate not in token_lengths:
                raise ValueError('Модель --compare відсутня у списку --sp або --tiktoken')
            pct = relative_reduction(token_lengths[base], token_lengths[candidate])
            group_file = groups.get(name)
            group_ids = read_groups(group_file, len(lines)) if group_file else None
            ci = (group_bootstrap_ci(token_lengths[base], token_lengths[candidate], group_ids, repeats, seed)
                  if group_ids is not None
                  else bootstrap_ci(token_lengths[base], token_lengths[candidate], repeats, seed))
            comparisons[name] = {'baseline': base, 'candidate': candidate,
                                 'token_reduction_pct': round(pct, 3),
                                 'bootstrap_95pct_ci': [round(ci[0], 3), round(ci[1], 3)],
                                 'bootstrap_resamples': repeats,
                                 'bootstrap_unit': 'document' if group_ids is not None else 'row',
                                 'document_groups': len(set(group_ids)) if group_ids else None,
                                 'warning': ('Cluster bootstrap by supplied document groups; still subject to corpus selection bias.'
                                             if group_ids else 'Row bootstrap assumes independent rows and may understate uncertainty.')}
    return metrics, comparisons


def render_md(result) -> str:
    rows = [
        '# UTP — звіт про токенізацію', '',
        '**Це вимірювання токенізації, а не швидкості, ціни або якості роботи мовної моделі.**', '',
        '| Корпус | Мова | Токенізатор | Токенів | Токенів/слово | Байтів/токен | Відновлення |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for name, corpus in result['corpora'].items():
        for tok, m in corpus['metrics'].items():
            rows.append(f'| {name} | {corpus["language"]} | {tok} | {m["tokens"]} | {m["tokens_per_word"]} | {m["bytes_per_token"]} | {m["roundtrip_errors"]} |')
    rows += ['', '## Парне порівняння на тих самих текстах', '']
    if not result['comparisons']:
        rows.append('Для порівняння додайте `--compare baseline,candidate`.')
    for name, comp in result['comparisons'].items():
        rows.append(f'- **{name}**: {comp["candidate"]} проти {comp["baseline"]}: '
                    f'**{comp["token_reduction_pct"]:+.2f}%** зменшення токенів; '
                    f'95% bootstrap інтервал [{comp["bootstrap_95pct_ci"][0]}, {comp["bootstrap_95pct_ci"][1]}]%, '
                    f'одиниця ресемплінгу: {comp["bootstrap_unit"]}')
    rows += ['', '## Умови інтерпретації', '',
             '- Різні розміри словників **не дозволяють** вважати різницю контрольованим порівнянням методів.',
             '- Для непаралельних корпусів порівняння української та англійської описове, а не семантично еквівалентне.',
             '- Якщо додано `--group`, довірчі інтервали розраховані ресемплінгом документів; інакше за рядками, що часто занижує похибку.',
             '- Нуль помилок відновлення показує лише коректність на цій вибірці.',
             '- Економія токенів у сторонньому токенізаторі **не впливає** на токени ChatGPT.', '']
    return '\n'.join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Порівняння токенізаторів на корпусах')
    parser.add_argument('--dataset', action='append', required=True, help='NAME:uk|en:PATH')
    parser.add_argument('--sp', action='append', default=[], help='NAME:MODEL_PATH')
    parser.add_argument('--tiktoken', nargs='+', default=[], help='напр. cl100k_base o200k_base')
    parser.add_argument('--compare', help='Пара baseline,candidate на ідентичних текстах')
    parser.add_argument('--group', action='append', default=[], help='NAME:PATH, документний ID для кожного рядка dataset')
    parser.add_argument('--out', type=Path, default=Path('runs/benchmark'))
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    datasets = [parse_spec(spec) for spec in args.dataset]
    if len({name for name, _, _ in datasets}) != len(datasets):
        parser.error('Назви корпусів мають бути унікальними')
    tokenizers = make_tokenizers(args.sp, args.tiktoken)
    compare = tuple(args.compare.split(',')) if args.compare else None
    if compare is not None and len(compare) != 2:
        parser.error('--compare має мати вигляд baseline,candidate')
    groups = dict(parse_model(x) for x in args.group)
    if len(groups) != len(args.group) or any(name not in {n for n,_,_ in datasets} for name in groups):
        parser.error('Назви груп повинні бути унікальними та відповідати --dataset')
    metrics, comparisons = evaluate(datasets, tokenizers, compare, args.bootstrap, args.seed, groups)
    result = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'note': 'Tokenizer-only evaluation. Not production LLM savings.',
        'corpora': metrics, 'comparisons': comparisons,
        'models': {name: {'vocab_size': spec[2], 'source': spec[3]} for name, spec in tokenizers.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    text = render_md(result)
    (args.out/'report.md').write_text(text+'\n', encoding='utf-8')
    print(text)
    print('\nФайли:', args.out.resolve())


if __name__ == '__main__':
    main()
