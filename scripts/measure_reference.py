#!/usr/bin/env python3
"""Descriptive reference token counts from real held-out text (no raw text published)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def measure(path: Path, names=("cl100k_base", "o200k_base")):
    import tiktoken
    rows = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise ValueError("Empty test corpus")
    out = {}
    for name in names:
        enc = tiktoken.get_encoding(name)
        tokens = sum(len(enc.encode(text, disallowed_special=())) for text in rows)
        errors = sum(enc.decode(enc.encode(text, disallowed_special=())) != text for text in rows)
        if errors:
            raise ValueError(f"Roundtrip loss for {name}: {errors}")
        out[name] = {"samples": len(rows), "tokens": tokens,
                     "tokens_per_sample": round(tokens/len(rows), 4),
                     "vocab_size": enc.n_vocab, "roundtrip_errors": errors}
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uk", type=Path, default=Path("data/ud_strict/uk_test.txt"))
    p.add_argument("--en", type=Path, default=Path("data/ud_strict/en_test.txt"))
    p.add_argument("--out", type=Path, default=Path("runs/ud_final/reference.json"))
    args = p.parse_args()
    result = {"note": "Different vocabulary sizes, contextual reference only; test sets are NOT translations.",
              "uk": measure(args.uk), "en": measure(args.en)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
