#!/usr/bin/env python3
"""v0.8: append Ukrainian-only BPE-like pieces to a *pretrained* tokenizer.

Preserves every existing token ID. Exercises a real open-weight SmolLM2 model;
publishes only aggregate token counts and bits per original UTF-8 byte.
This CC0-only engineering pilot is not independent external LLM validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from external_cc0 import SOURCES, download, load_cc0, normalize
from llm_adapter import (DEFAULT_MODEL, DEFAULT_REVISION, choose_paired_texts,
                         disjoint_cc0, ids_source, score_sentences)
from llm_continuation import training_schedule

CYRILLIC_WORD = re.compile(r"[А-Яа-яҐґЄєІіЇї]{4,}")
STEPS = (0, 64, 256)


def choose_ukrainian_pieces(training_texts, tokenizer, count=128):
    """Count Cyrillic substrings on CC0 train only; no heldout lookups."""
    if count < 1:
        raise ValueError("Need at least one new token")
    counter = Counter()
    for sentence in training_texts:
        for match in CYRILLIC_WORD.finditer(sentence):
            word = match.group(0)
            substrings = {
                word[i:i+size] for size in range(4, min(len(word), 10) + 1)
                for i in range(len(word)-size+1)
            }
            counter.update(substrings)
    ranked = sorted(
        (piece for piece, hits in counter.items()
         if hits >= 3 and
         len(tokenizer.encode(piece, add_special_tokens=False)) >= 2),
        key=lambda p: (-counter[p] * (len(p)-2), -len(p), p)
    )
    return ranked[:count]


def add_pieces(tokenizer, pieces):
    from tokenizers import AddedToken
    if not pieces or len(pieces) != len(set(pieces)):
        raise ValueError("No unique pieces")
    if any(not re.fullmatch(r"[А-Яа-яҐґЄєІіЇї]+", p) for p in pieces):
        raise ValueError("Only Ukrainian Cyrillic pieces can be appended")
    old_vocab_size = len(tokenizer)
    before = [tokenizer.convert_ids_to_tokens(i) for i in range(old_vocab_size)]
    added = tokenizer.add_tokens([
        AddedToken(p, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for p in pieces
    ])
    if added != len(pieces):
        raise ValueError(f"Expected {len(pieces)} new IDs, added {added}")
    if len(tokenizer) != old_vocab_size + added:
        raise ValueError("Unexpected new vocabulary size")
    if before != [tokenizer.convert_ids_to_tokens(i) for i in range(old_vocab_size)]:
        raise ValueError("An original pretrained token ID changed")
    for i, piece in enumerate(pieces):
        if tokenizer.convert_tokens_to_ids(piece) != old_vocab_size + i:
            raise ValueError("Added IDs not sequential or altered")
    return old_vocab_size


def verify_old_english_ids(original, augmented, english):
    """On completely Cyrillic-free English strings, IDs must not change."""
    count = 0
    for text in english:
        if re.search(r"[\u0400-\u052f]", text):
            continue
        if original.encode(text, add_special_tokens=False) != augmented.encode(
                text, add_special_tokens=False):
            raise ValueError("English tokenization changed despite Cyrillic-only pieces")
        count += 1
    if count < 30:
        raise ValueError("Insufficient pure-English invariant checks")
    return count


def initialize_appended_embeddings(model, original_tokenizer, pieces, old_size):
    """Initialize only the appended rows; leave original vectors byte-identical."""
    import torch
    emb = model.get_input_embeddings().weight.detach().clone()
    head = model.get_output_embeddings().weight.detach().clone()
    if emb.shape[0] != old_size or head.shape[0] != old_size:
        raise ValueError("Old vocabulary size does not match model embeddings")
    input_new, output_new = [], []
    for piece in pieces:
        ids = original_tokenizer.encode(piece, add_special_tokens=False)
        if not ids:
            raise ValueError(f"No original encoding for {piece!r}")
        input_new.append(emb[ids].mean(dim=0))
        output_new.append(head[ids].mean(dim=0))
    with torch.no_grad():
        model.resize_token_embeddings(old_size + len(pieces), mean_resizing=False)
        input_emb = model.get_input_embeddings().weight
        output_emb = model.get_output_embeddings().weight
        if not torch.equal(input_emb[:old_size], emb):
            raise ValueError("Original input embeddings changed on resizing")
        if not torch.equal(output_emb[:old_size], head):
            raise ValueError("Original output embeddings changed on resizing")
        input_emb[old_size:] = torch.stack(input_new)
        output_emb[old_size:] = torch.stack(output_new)
    return {"old_vocab": old_size, "added": len(pieces), "new_vocab": old_size + len(pieces),
            "original_embeddings_unchanged_before_training": True}


def only_new_rows_optimizer(model, old_size, lr=0.0004):
    import torch
    if lr <= 0:
        raise ValueError("Invalid learning rate")
    for param in model.parameters():
        param.requires_grad_(False)
    unique = []
    seen = set()
    for embedding in (model.get_input_embeddings(), model.get_output_embeddings()):
        weight = embedding.weight
        if id(weight) in seen:
            continue
        seen.add(id(weight))
        weight.requires_grad_(True)
        weight.register_hook(lambda grad, old=old_size:
                             torch.cat((torch.zeros_like(grad[:old]), grad[old:]), dim=0))
        unique.append(weight)
    return torch.optim.AdamW(unique, lr=lr, weight_decay=0.0)


def train_interval(model, tokenizer, optimizer, samples):
    import torch
    import torch.nn.functional as F
    model.train()
    losses, done = [], {"uk": 0, "en": 0}
    for language, text in samples:
        ids = ids_source(tokenizer, text)
        if len(ids) < 3 or len(ids) > 82:
            raise ValueError("Training example outside prescribed length cap")
        input_ids = torch.tensor([ids], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                               input_ids[:, 1:].reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite adaptation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        done[language] += 1
        losses.append(float(loss.detach()))
    model.eval()
    return {"steps": sum(done.values()), "uk_steps": done["uk"], "en_steps": done["en"],
            "mean_loss_per_training_token": round(sum(losses) / len(losses), 4)}


def invariant_original_rows(model, originals, old_size):
    import torch
    for embedding, snapshot in zip((model.get_input_embeddings(), model.get_output_embeddings()), originals):
        if not torch.equal(embedding.weight[:old_size].detach(), snapshot):
            raise ValueError("Forbidden changes to old token embeddings")


def compare(model, paired, tokenizer):
    return {language: score_sentences(model, texts, lambda s: ids_source(tokenizer, s))
            for language, texts in paired.items()}


def assert_paired(results, requested):
    for label, score in results.items():
        for lang, stats in score.items():
            if stats["texts_evaluated"] != requested[lang]["count"] or stats["bytes"] != requested[lang]["bytes"]:
                raise ValueError(f"Unpaired evaluation: {label}/{lang}")


def experiment(count=128, samples=30, seed=42, model_id=DEFAULT_MODEL,
               revision=DEFAULT_REVISION):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if samples < 10 or samples > 50:
        raise ValueError("Use 10..50 paired evaluation sentences per language")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    training, external, sources = {}, {}, {}
    for lang, limit in (("uk", 3000), ("en", 2000)):
        texts, info = load_cc0(download(SOURCES[lang]), lang, set(), limit)
        training[lang], external[lang] = disjoint_cc0(texts)
        sources[lang] = {"url": SOURCES[lang], "sha256": info["compressed_sha256"],
                         "license": "CC0 sentence export",
                         "train_count": len(training[lang]), "test_count": len(external[lang])}
    original = AutoTokenizer.from_pretrained(model_id, revision=revision)
    augmented = AutoTokenizer.from_pretrained(model_id, revision=revision)
    pieces = choose_ukrainian_pieces(training["uk"], original, count=count)
    if len(pieces) < min(20, count):
        raise ValueError(f"Insufficient candidate pieces from small CC0 sample: {len(pieces)}")
    old_size = add_pieces(augmented, pieces)
    english_preserved = verify_old_english_ids(original, augmented,
                                               training["en"] + external["en"])
    check_roundtrip = lambda t: augmented.decode(
        augmented.encode(t, add_special_tokens=False),
        skip_special_tokens=False, clean_up_tokenization_spaces=False
    ) == t
    for lang in ("uk", "en"):
        bad = [s for s in training[lang][:30] + external[lang][:30] if not check_roundtrip(s)]
        if bad:
            raise ValueError(f"New tokenizer failed exact Unicode roundtrip: {lang}, {len(bad)}")
    # Fix pairs before any evaluation, on the same original UTF-8 text.
    paired = {
        lang: choose_paired_texts(
            [s for s in external[lang] if check_roundtrip(s)],
            lambda s: ids_source(original, s),
            lambda s: ids_source(augmented, s), samples
        ) for lang in ("uk", "en")
    }
    eligible = {
        lang: [s for s in training[lang] if check_roundtrip(s) and
               3 <= len(ids_source(augmented, s)) <= 82]
        for lang in ("uk", "en")
    }
    if any(len(eligible[lang]) < 30 for lang in ("uk", "en")):
        raise ValueError("Insufficient bilingual CC0 training examples")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float32
    ).to("cpu").eval()
    initial_sha = getattr(model.config, "_commit_hash", None)
    if initial_sha is not None and initial_sha != revision:
        raise ValueError("Unexpected model revision")
    initial = compare(model, paired, original)
    init = initialize_appended_embeddings(model, original, pieces, old_size)
    originals = (model.get_input_embeddings().weight[:old_size].detach().clone(),
                 model.get_output_embeddings().weight[:old_size].detach().clone())
    results = {"0": compare(model, paired, augmented)}
    optimizer = only_new_rows_optimizer(model, old_size)
    schedule = list(training_schedule(eligible["uk"], eligible["en"], STEPS[-1], seed=seed))
    progress = {}
    current = 0
    for target in STEPS[1:]:
        progress[str(target)] = train_interval(model, augmented, optimizer,
                                               schedule[current:target])
        invariant_original_rows(model, originals, old_size)
        results[str(target)] = compare(model, paired, augmented)
        current = target
    all_scores = {"original": initial, **{f"step_{k}": v for k, v in results.items()}}
    sample_meta = {
        lang: {"count": len(paired[lang]), "bytes": len("".join(paired[lang]).encode("utf-8")),
               "sha256": hashlib.sha256("\n".join(paired[lang]).encode("utf-8")).hexdigest()}
        for lang in ("uk", "en")
    }
    assert_paired(all_scores, sample_meta)
    return {
        "experiment": "v08_append_cyrillic_to_original_tokenizer",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id, "pinned_revision": revision, "model_revision": initial_sha,
        "token_additions": init,
        "piece_selection": {"source": "CC0 TRAIN Ukrainian only",
                            "algorithm": "Cyrillic 4-10 character substrings with train freq>=3",
                            "pieces_sha256": hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()},
        "english_token_id_invariance_checked": english_preserved,
        "data_sources": sources, "paired_samples": sample_meta,
        "adaptation": {"optimizer": "AdamW", "lr": 0.0004, "trainable": "only added input/output rows",
                       "preserved_old_embeds_after_each_checkpoint": True,
                       "train_mix": "3 UA examples : 1 EN example",
                       "checkpoints": list(STEPS), "intervals": progress},
        "measurement": "negative-log2-likelihood per original UTF-8 byte (lower better)",
        "original_model": initial, "augmented_model": results,
        "limitations": [
            "Exploratory engineering test; Tatoeba source previously evaluated in UTP v0.5-v0.7.",
            "Only new rows were updated; other transformer and original embedding weights frozen.",
            "Adding output vocabulary rows changes softmax even when old input IDs and embeddings are unchanged.",
            "30 sentences per language, same source, no independent quality/domain/task test.",
            "Tokenizer pieces selected on CC0 TRAIN only, not on heldout token counts.",
            "Training checkpoints are repeated looks at same heldout sample; no checkpoint selected by results.",
            "GitHub CPU runtime is not a benchmark of production latency.",
            "No relation to ChatGPT or GPT-6 token cost; no model weights or corpora redistributed."
        ]
    }


def markdown(report):
    o, outcomes = report["original_model"], report["augmented_model"]
    lines = ["# UTP v0.8 — append Ukrainian pieces, preserve the original pretrained vocabulary",
             "", "**Engineering pilot with a very small CC0 corpus, not a deployment result.**", "",
             f"Original model: {report['model_id']} at {report['pinned_revision']}.",
             f"Old tokens: {report['token_additions']['old_vocab']}; appended: "
             f"{report['token_additions']['added']}; total: {report['token_additions']['new_vocab']}.",
             f"English strings with identical IDs: {report['english_token_id_invariance_checked']}.", "",
             "| Condition | UA tokens | UA bits/UTF-8 byte | EN tokens | EN bits/UTF-8 byte |",
             "|---|---:|---:|---:|---:|"]
    def add_row(label, value):
        u, e = value["uk"], value["en"]
        lines.append(f"| {label} | {u['text_tokens']} | {u['bits_per_utf8_byte']} | "
                     f"{e['text_tokens']} | {e['bits_per_utf8_byte']} |")
    add_row("Original tokenizer + pretrained model", o)
    for step, value in outcomes.items():
        add_row(f"Appended Ukrainian tokens; {step} new-row train steps", value)
    lines += ["", "## Limitations", ""]
    lines.extend("- "+x for x in report["limitations"])
    return "\n".join(lines) + "\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/additive_v08.json"))
    args=p.parse_args()
    result=experiment(count=args.new_tokens, samples=args.samples, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__=="__main__":
    main()
