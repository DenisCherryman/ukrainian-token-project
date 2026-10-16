#!/usr/bin/env python3
"""v0.7 exploratory continued adaptation of a real LLM with a CC0 tokenizer.

All conditions use identical paired evaluation sentences and the same pinned
open-weight model. The 0/16/64/128 checkpoints are exploratory, not selected
using an independent test. No weights or third-party sentences are published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from external_cc0 import SOURCES, download, load_cc0, normalize
from llm_adapter import (
    DEFAULT_MODEL, DEFAULT_REVISION, choose_paired_texts, disjoint_cc0,
    ids_new, ids_source, replace_embeddings, score_sentences,
    train_cc0_tokenizer,
)

CHECKPOINTS = (0, 16, 64, 128)
BATCH_MAX_IDS = 66
UK_TRAIN_SHARE = 0.75


def training_schedule(uk: list[str], en: list[str], steps: int, seed: int = 42):
    """Deterministically cycle independently shuffled train samples, 3 UA : 1 EN.

    The ratio here is a ratio of *steps*, not UTF-8 bytes or equal-length tokens.
    """
    if not uk or not en or steps < 0:
        raise ValueError("Nonempty bilingual pools and nonnegative steps required")
    rng = random.Random(seed)
    pools = {"uk": list(uk), "en": list(en)}
    for values in pools.values():
        rng.shuffle(values)
    positions = {"uk": 0, "en": 0}
    for step in range(steps):
        lang = "uk" if step % 4 != 3 else "en"
        pool = pools[lang]
        at = positions[lang]
        if at == len(pool):
            rng.shuffle(pool)
            at = 0
        positions[lang] = at + 1
        yield lang, pool[at]


def eligible_training(uk: list[str], en: list[str], encode, limit=BATCH_MAX_IDS):
    def clean(sentences):
        return [s for s in sentences if 3 <= len(encode(s)) <= limit]
    out_uk, out_en = clean(uk), clean(en)
    if len(out_uk) < 20 or len(out_en) < 20:
        raise ValueError(f"Insufficient small-CPU-sequence training data: {len(out_uk)} UK, {len(out_en)} EN")
    return out_uk, out_en


def paired_scores(model, texts, original_tokenizer, sp, max_tokens=80):
    encode = lambda t: ids_new(sp, t)
    return {
        lang: score_sentences(model, values, encode, max_text_tokens=max_tokens)
        for lang, values in texts.items()
    }


def make_optimizer(model, lr=3e-4):
    import torch
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    model.get_output_embeddings().weight.requires_grad_(True)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("Expected trainable embedding parameters")
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


def train_increment(model, sp, optimizer, batches, max_step):
    """Train a predetermined number of bilingual examples, including EOS."""
    import torch
    import torch.nn.functional as F
    model.train()
    n, counts, loss_sum = 0, {"uk": 0, "en": 0}, 0.
    for lang, text in batches:
        ids = ids_new(sp, text)
        if not (3 <= len(ids) <= BATCH_MAX_IDS):
            raise ValueError("Training input no longer meets eligibility rule")
        inputs = torch.tensor([ids], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=inputs, use_cache=False).logits[:, :-1, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               inputs[:, 1:].reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        n += 1
        counts[lang] += 1
        loss_sum += float(loss.detach())
    if n != max_step:
        raise ValueError(f"Expected {max_step} new training steps, got {n}")
    model.eval()
    return {"steps": n, "uk_steps": counts["uk"], "en_steps": counts["en"],
            "mean_train_loss_per_token": round(loss_sum / n, 4) if n else None}


def verify_paired(reference, all_results, holdout):
    """Refuse invalid comparisons if scores were computed on other bytes."""
    for lang, texts in holdout.items():
        wanted = len(texts)
        reference_stats = reference[lang]
        if reference_stats["texts_evaluated"] != wanted:
            raise ValueError("Original model evaluated a different number of sentences")
        for label, by_lang in all_results.items():
            trial = by_lang[lang]
            if trial["texts_evaluated"] != wanted:
                raise ValueError(f"Unpaired sentence selection at {label}/{lang}")
            if trial["bytes"] != reference_stats["bytes"]:
                raise ValueError(f"Unpaired byte denominator at {label}/{lang}")


def run(steps=CHECKPOINTS, sample_count=30, seed=42, model_id=DEFAULT_MODEL,
        revision=DEFAULT_REVISION, vocab=1024):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tuple(steps) != CHECKPOINTS:
        raise ValueError("Checkpoint schedule was preregistered: 0/16/64/128")
    if sample_count < 10 or sample_count > 50:
        raise ValueError("Use 10-50 paired CC0 sentences per language")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    training, evaluation, provenance = {}, {}, {}
    for lang, limit in (("uk", 3000), ("en", 2000)):
        texts, info = load_cc0(download(SOURCES[lang]), lang, set(), limit)
        train, heldout = disjoint_cc0(texts)
        train_keys = {normalize(x) for x in train}
        if train_keys & {normalize(x) for x in heldout}:
            raise ValueError("CC0 train/test contamination")
        training[lang], evaluation[lang] = train, heldout
        provenance[lang] = {
            "source": SOURCES[lang],
            "compressed_sha256": info["compressed_sha256"],
            "license": "Tatoeba CC0 sentence export",
            "training_sentences": len(train), "holdout_sentences": len(heldout),
        }
    work = Path("runs/llm_continuation_v07")
    sp, tokenizer_training = train_cc0_tokenizer(training["uk"], training["en"], work, vocab)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    encode_orig = lambda t: ids_source(pretrained_tokenizer, t)
    encode_candidate = lambda t: ids_new(sp, t)
    paired = {
        lang: choose_paired_texts(texts, encode_orig, encode_candidate, sample_count)
        for lang, texts in evaluation.items()
    }
    selected_keys = {lang: {normalize(x) for x in texts} for lang, texts in paired.items()}
    for lang in ("uk", "en"):
        if selected_keys[lang] & {normalize(x) for x in training[lang]}:
            raise ValueError("Paired evaluation training overlap")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float32
    ).to("cpu")
    revision_observed = getattr(model.config, "_commit_hash", None)
    if revision_observed is not None and revision_observed != revision:
        raise ValueError("Model revision does not match pinned SHA")
    baseline = {
        lang: score_sentences(model, texts, encode_orig)
        for lang, texts in paired.items()
    }
    replace_embeddings(model, sp, pretrained_tokenizer)
    result = {"0": paired_scores(model, paired, pretrained_tokenizer, sp)}
    optimizer = make_optimizer(model)
    filtered_uk, filtered_en = eligible_training(
        training["uk"], training["en"], encode_candidate
    )
    # Schedule is seeded independently, never using CC0 test or its scores.
    batches = list(training_schedule(filtered_uk, filtered_en, CHECKPOINTS[-1], seed))
    training_log = {}
    previous = 0
    for target in CHECKPOINTS[1:]:
        subset = batches[previous:target]
        training_log[str(target)] = train_increment(model, sp, optimizer, subset, target - previous)
        result[str(target)] = paired_scores(model, paired, pretrained_tokenizer, sp)
        previous = target
    verify_paired(baseline, result, paired)
    return {
        "experiment": "v07_bilingual_embedding_continuation_engineering",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_id, "model_revision": revision,
        "model_revision_observed": revision_observed,
        "tokenizer_training": tokenizer_training,
        "data": provenance,
        "paired_test": {
            "sha256": {lang: hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
                       for lang, texts in paired.items()},
            "counts": {lang: len(texts) for lang, texts in paired.items()},
            "selection": "same sentences under both tokenizers, selected before scoring",
        },
        "training": {
            "trainable": "input/output token embeddings only",
            "optimizer": "AdamW", "learning_rate": 0.0003,
            "selection": "fixed checkpoint schedule, no heldout-driven selection",
            "training_step_mix": "3 Ukrainian : 1 English (examples, not bytes)",
            "eligible_train_sentences": {"uk": len(filtered_uk), "en": len(filtered_en)},
            "step_intervals": training_log,
        },
        "metrics": "negative-log2-likelihood bits per original UTF-8 byte",
        "original": baseline,
        "new_tokenizer_checkpoints": result,
        "limitations": [
            "Engineering smoke test using a small CC0 dataset; no newly independent domain or downstream task test.",
            "The same source (Tatoeba) was used in v0.5 and v0.6 exploratory work.",
            "Checkpoints are multiple looks at the same heldout texts, not independent confirmatory evaluations.",
            "Only embedding tables were updated; the transformer backbone remained frozen.",
            "The 1024-token new vocabulary differs in size from the original model tokenizer.",
            "No licensed/restricted UD text or UD-trained model is used in this workflow.",
            "Training corpus is too small to demonstrate model-quality recovery or generalizable improvements.",
            "Raw forward pass time is not hardware-normalized production speed.",
            "No ChatGPT/GPT-6 token-price or quality improvement implied.",
        ],
    }


def render(report):
    lines = [
        "# UTP v0.7 — bilingual embedding continuation, open-weight LLM", "",
        "**Exploratory engineering test. Lower bits/UTF-8 byte = better prediction.**", "",
        f"Model: {report['model']} ({report['model_revision']})", "",
        "| Condition | Ukrainian bits/byte | UA text tokens | English bits/byte | EN text tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    base = report["original"]
    lines.append(f"| Original pretrained tokenizer | {base['uk']['bits_per_utf8_byte']} | "
                 f"{base['uk']['text_tokens']} | {base['en']['bits_per_utf8_byte']} | "
                 f"{base['en']['text_tokens']} |")
    for step, scores in report["new_tokenizer_checkpoints"].items():
        lines.append(f"| New 1024 vocabulary; {step} embedding steps | "
                     f"{scores['uk']['bits_per_utf8_byte']} | {scores['uk']['text_tokens']} | "
                     f"{scores['en']['bits_per_utf8_byte']} | {scores['en']['text_tokens']} |")
    lines += ["", "## Caveats", ""]
    lines.extend("- " + limitation for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/llm_continuation_v07.json"))
    args = p.parse_args()
    report = run(sample_count=args.samples, seed=args.seed, vocab=args.vocab)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(render(report))


if __name__ == "__main__":
    main()
