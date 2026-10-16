#!/usr/bin/env python3
"""Ukrainian Token Project: reproducible, budget-matched tokenizer pilot.

Runs on local, properly licensed UTF-8 text files (one sample per line).
The two train sets have approximately equal raw UTF-8 byte budgets, and
both tokenizers have the same vocabulary size and normalization settings.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError as exc:
    raise SystemExit('Встановіть залежності: pip install -r requirements.txt') from exc

WORD_RE = re.compile(r"\w+(?:[’'\-]\w+)*", re.UNICODE)


def read_lines(path: Path) -> list[str]:
    lines = []
    seen = set()
    for text in path.read_text(encoding='utf-8-sig').splitlines():
        text = text.strip('\r\n')
        if not text.strip():
            continue
        # Deduplicate without changing the submitted string or its orthography.
        key = text.casefold().strip()
        if key not in seen:
            lines.append(text)
            seen.add(key)
    if len(lines) < 10:
        raise ValueError(f'{path}: потрібно хоча б 10 унікальних рядків')
    return lines


def split_by_hash(lines: list[str], test_pct: int, seed: int) -> tuple[list[str], list[str]]:
    """Stable train/test split independent of input line order."""
    train, test = [], []
    for line in lines:
        digest = hashlib.sha256(f'{seed}:'.encode() + line.casefold().strip().encode('utf-8')).digest()
        group = int.from_bytes(digest[:8], 'big') % 100
        (test if group < test_pct else train).append(line)
    if not train or not test:
        raise ValueError('Недостатньо даних для розподілу train/test; додайте рядки.')
    return train, test


def line_bytes(lines: list[str]) -> int:
    return sum(len((line+'\n').encode('utf-8')) for line in lines)


def select_to_budget(lines: list[str], cap: int, seed: int) -> list[str]:
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    output = []
    current = 0
    for line in shuffled:
        size = len((line+'\n').encode('utf-8'))
        if current + size <= cap:
            output.append(line)
            current += size
    if not output:
        raise ValueError('Замалий бюджет для хоча б одного навчального речення.')
    return output


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(s+'\n' for s in lines), encoding='utf-8')


def train_tokenizer(input_path: Path, prefix: Path, vocab: int) -> Path:
    # Identity prevents silently normalizing away Ukrainian orthographic distinctions.
    # Byte fallback handles code points absent from training, including emojis.
    spm.SentencePieceTrainer.train(
        input=str(input_path), model_prefix=str(prefix),
        model_type='bpe', vocab_size=vocab, byte_fallback=True,
        character_coverage=1.0, hard_vocab_limit=False,
        normalization_rule_name='identity',
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0,
        shuffle_input_sentence=False, num_threads=2,
        max_sentence_length=16384,
    )
    return prefix.with_suffix('.model')


def summary_sentencepiece(model_path: Path, lines: list[str]) -> dict:
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    return _summary(lines, processor.encode, processor.decode, processor.vocab_size())


def _summary(lines, encode, decode, vocab_size) -> dict:
    token_total = 0
    errors = 0
    word_total = 0
    byte_total = 0
    for line in lines:
        ids = encode(line)
        token_total += len(ids)
        word_total += len(WORD_RE.findall(line))
        byte_total += len(line.encode('utf-8'))
        if decode(ids) != line:
            errors += 1
    return {
        'lines': len(lines), 'tokens': token_total, 'words': word_total,
        'utf8_bytes': byte_total, 'vocab_size': vocab_size,
        'tokens_per_word': round(token_total / word_total, 4) if word_total else None,
        'bytes_per_token': round(byte_total / token_total, 4) if token_total else None,
        'round_trip_errors': errors,
    }


def tiktoken_summary(lines: list[str], name: str) -> dict | None:
    try:
        import tiktoken
    except ImportError:
        return None
    encoding = tiktoken.get_encoding(name)
    return _summary(lines, encoding.encode, encoding.decode, encoding.n_vocab)


def main() -> None:
    parser = argparse.ArgumentParser(description='Ukrainian Token Project: pilot experiment')
    parser.add_argument('--uk', type=Path, required=True, help='Український корпус, UTF-8, 1 рядок = 1 приклад')
    parser.add_argument('--en', type=Path, required=True, help='Англійський корпус для багатомовного порівняння')
    parser.add_argument('--uk-test', type=Path, help='Незалежний офіційний тестовий набір (не розділяти --uk)')
    parser.add_argument('--en-test', type=Path, help='Незалежний англійський тест (потрібен разом із --uk-test)')
    parser.add_argument('--out', type=Path, default=Path('runs/pilot'))
    parser.add_argument('--vocab', type=int, default=1024)
    parser.add_argument('--train-budget-bytes', type=int, default=10_000_000)
    parser.add_argument('--test-percent', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--openai-encoding', default='o200k_base',
                        help='Лише додатковий орієнтир; НЕ рівний за розміром словника')
    args = parser.parse_args()
    if not (5 <= args.test_percent <= 50):
        parser.error('--test-percent має бути між 5 та 50')
    if args.vocab < 384:
        parser.error('--vocab має бути >= 384 (256 byte fallback + українські символи)')

    if bool(args.uk_test) != bool(args.en_test):
        parser.error('--uk-test та --en-test потрібно вказувати разом')
    if args.uk_test:
        # Honor official upstream splits, never reshuffle a held-out test into training.
        uk_train, uk_test = read_lines(args.uk), read_lines(args.uk_test)
        en_train, en_test = read_lines(args.en), read_lines(args.en_test)
        split_mode = 'external_heldout'
    else:
        uk_train, uk_test = split_by_hash(read_lines(args.uk), args.test_percent, args.seed)
        en_train, en_test = split_by_hash(read_lines(args.en), args.test_percent, args.seed)
        split_mode = 'sentence_hash_pilot'
    for language, train, test in (('uk', uk_train, uk_test), ('en', en_train, en_test)):
        overlap = {s.casefold().strip() for s in train} & {s.casefold().strip() for s in test}
        if overlap:
            example = next(iter(overlap))
            raise ValueError(f'{language}: {len(overlap)} рядків збігаються між train/test. '
                             f'Усуньте витік даних (приклад: {example[:80]!r}).')


    # Equal (approximately) UTF-8 budgets for monolingual and bilingual training.
    # Half of multilingual training bytes come from each language.
    budget = min(args.train_budget_bytes, line_bytes(uk_train), 2 * line_bytes(en_train))
    if budget < 2048:
        raise ValueError('Замало навчальних даних: потрібно щонайменше 2048 байтів.')
    mono = select_to_budget(uk_train, budget, args.seed)
    balanced_uk = select_to_budget(uk_train, budget // 2, args.seed + 1)
    balanced_en = select_to_budget(en_train, budget // 2, args.seed + 2)
    balanced = balanced_uk + balanced_en
    random.Random(args.seed).shuffle(balanced)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_lines(out / 'train_uk_only.txt', mono)
    write_lines(out / 'train_balanced.txt', balanced)
    write_lines(out / 'test_uk.txt', uk_test)
    write_lines(out / 'test_en.txt', en_test)

    model_uk = train_tokenizer(out / 'train_uk_only.txt', out / 'uk_only', args.vocab)
    model_mixed = train_tokenizer(out / 'train_balanced.txt', out / 'balanced_uk_en', args.vocab)

    metrics = {
        'uk_only': {'uk': summary_sentencepiece(model_uk, uk_test),
                    'en': summary_sentencepiece(model_uk, en_test)},
        'balanced_uk_en': {'uk': summary_sentencepiece(model_mixed, uk_test),
                           'en': summary_sentencepiece(model_mixed, en_test)},
    }
    actual_a = metrics['uk_only']['uk']['vocab_size']
    actual_b = metrics['balanced_uk_en']['uk']['vocab_size']
    if actual_a != args.vocab or actual_b != args.vocab:
        raise ValueError(
            f'Фактичні словники {actual_a} і {actual_b}; зменште --vocab, щоб порівнювати рівні словники.'
        )
    tiktoken_uk = tiktoken_summary(uk_test, args.openai_encoding)
    tiktoken_en = tiktoken_summary(en_test, args.openai_encoding)
    if tiktoken_uk is not None:
        metrics[f'openai_{args.openai_encoding}_reference_only'] = {
            'uk': tiktoken_uk, 'en': tiktoken_en,
        }

    baseline = metrics['balanced_uk_en']['uk']['tokens']
    ukrainian = metrics['uk_only']['uk']['tokens']
    saving = 100 * (baseline - ukrainian) / baseline
    result = {
        'note': 'Pilot only; NOT representative of GPT-6 or production savings.',
        'setup': {
            'seed': args.seed, 'split_mode': split_mode, 'test_percent': args.test_percent,
            'requested_vocab': args.vocab, 'train_budget_bytes': budget,
            'actual_uk_train_bytes': line_bytes(mono),
            'actual_balanced_train_bytes': line_bytes(balanced),
            'balanced_uk_bytes': line_bytes(balanced_uk),
            'balanced_en_bytes': line_bytes(balanced_en),
            'uk_test_lines': len(uk_test), 'en_test_lines': len(en_test),
        },
        'uk_only_relative_token_reduction_pct': round(saving, 2),
        'metrics': metrics,
    }
    (out / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    md = ['# Результати UTP', '',
          '**Увага:** це пілот, а не заявлена економія в ChatGPT або GPT-6.', '',
          f'Розмір словника моделей SentencePiece: {args.vocab}',
          f'Бюджет навчання (UTF-8): {budget:,} байтів; тестування на відкладених рядках.', '',
          '| Токенізатор | Українська: токени/слово | Англійська: токени/слово | Помилки відновлення UA/EN |',
          '|---|---:|---:|---:|']
    for name, pairs in metrics.items():
        md.append(f'| {name} | {pairs["uk"]["tokens_per_word"]} | {pairs["en"]["tokens_per_word"]} | {pairs["uk"]["round_trip_errors"]}/{pairs["en"]["round_trip_errors"]} |')
    md.extend(['', f'Зміна токенів на українському тесті (uk_only проти balanced): **{saving:+.2f}% скорочення**.',
               'Додатний результат означає меншу кількість токенів. Від’ємний — більшу.',
               '', '## Обмеження',
               '- Рівний розмір словника лише для двох навчених SentencePiece-моделей.',
               '- `o200k_base` — додатковий орієнтир з іншим розміром словника, не чесне порівняння якості.',
               '- Навчальні вибірки приблизно, а не точно зрівняні за UTF-8 байтами.',
               '- Відсутність помилок відновлення на тесті не доводить відсутності помилок для всіх текстів.',
               '- Навіть значне скорочення токенів не гарантує кращої роботи мовної моделі.',
               '- Для наукової оцінки потрібні великі різнодоменні корпуси й повтори експериментів.', ''])
    (out / 'results.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))
    print(f'\nАртефакти: {out.resolve()}')


if __name__ == '__main__':
    main()
