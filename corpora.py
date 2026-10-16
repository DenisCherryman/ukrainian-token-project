#!/usr/bin/env python3
"""Fetch publicly hosted UD datasets for locally run, non-commercial research.

Original CoNLL-U corpora are NOT redistributed in this repository.
Respect dataset licenses, especially Ukrainian CC BY-NC-SA 4.0 and
source-content copyrights noted by the English dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPOS = {
    'uk': ('UD_Ukrainian-IU', 'https://creativecommons.org/licenses/by-nc-sa/4.0/', 'CC BY-NC-SA 4.0; non-commercial only'),
    'en': ('UD_English-EWT', 'https://creativecommons.org/licenses/by-sa/4.0/', 'CC BY-SA 4.0 for annotations; original source text may have other rights'),
}
BASE = 'https://raw.githubusercontent.com/UniversalDependencies'


def extract_text(conllu: str) -> list[str]:
    """Read literal # text= fields; never reconstruct from syntactic token rows."""
    prefix = '# text = '
    texts = [line[len(prefix):] for line in conllu.splitlines() if line.startswith(prefix)]
    return [text for text in texts if text.strip()]


def build(out: Path, timeout: float=60, refs: dict | None=None) -> dict:
    manifest = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'corpora': {}}
    out.mkdir(parents=True, exist_ok=True)
    refs = refs or {'uk': 'master', 'en': 'master'}
    for lang, (repo, license_url, license_note) in REPOS.items():
        manifest['corpora'][lang] = {'source': f'https://github.com/UniversalDependencies/{repo}',
                                    'license_url': license_url, 'license_note': license_note, 'files': {}}
        for split in ('train', 'dev', 'test'):
            filename = f'{lang}_{"iu" if lang == "uk" else "ewt"}-ud-{split}.conllu'
            url = f'{BASE}/{repo}/{refs[lang]}/{filename}'
            request = urllib.request.Request(url, headers={'User-Agent': 'UTP-noncommercial-research/0.2'})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
            except Exception as exc:
                raise RuntimeError(f'Не вдалося завантажити {url}: {exc}\nПеревірте доступ до GitHub; можна повторити команду пізніше.') from exc
            conllu_path = out / filename
            conllu_path.write_bytes(raw)
            decoded = raw.decode('utf-8-sig')
            rows = extract_text(decoded)
            if not rows:
                raise ValueError(f'Немає # text у {url}')
            path = out / f'{lang}_{split}.txt'
            path.write_text(''.join(row+'\n' for row in rows), encoding='utf-8')
            manifest['corpora'][lang]['files'][split] = {
                'source_url': url, 'upstream_sha256': hashlib.sha256(raw).hexdigest(),
                'line_count': len(rows), 'local_file': str(path),
                'conllu_file': str(conllu_path), 'upstream_ref': refs[lang],
            }
            print(f'{lang}/{split}: {len(rows)} речень')
    (out/'SOURCES.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Завантаження UD лише для локального аналізу')
    parser.add_argument('--out', type=Path, default=Path('data/ud'))
    parser.add_argument('--accept-license', action='store_true', help='Підтверджую дотримання умов ліцензій UD')
    parser.add_argument('--uk-ref', default='master', help='Фіксований git SHA бажаний замість master')
    parser.add_argument('--en-ref', default='master', help='Фіксований git SHA бажаний замість master')
    args = parser.parse_args()
    if not args.accept_license:
        parser.error('Для завантаження ознайомтеся з ліцензіями у README і вкажіть --accept-license')
    build(args.out, refs={'uk': args.uk_ref, 'en': args.en_ref})


if __name__ == '__main__':
    main()
