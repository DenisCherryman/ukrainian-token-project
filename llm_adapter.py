#!/usr/bin/env python3
"""v0.6 engineering demonstration: transplant a CC0-trained tokenizer into an LLM.

Not a model-quality benchmark. Reports bits per original UTF-8 byte before and
after optional *embedding-only* continuation, never perplexity/token across
different vocabularies. No datasets or model weights are uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from external_cc0 import SOURCES, download, load_cc0, normalize
from utp import line_bytes, select_to_budget, write_lines

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def disjoint_cc0(texts: list[str], train_pct: int = 75) -> tuple[list[str], list[str]]:
    if not (20 <= train_pct <= 90):
        raise ValueError("train_pct outside 20..90")
    training, evaluation = [], []
    for text in texts:
        # Stable text-hash partition: export order does not affect membership.
        bucket = int(hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:12], 16) % 100
        (training if bucket < train_pct else evaluation).append(text)
    if not training or not evaluation:
        raise ValueError("Need train and evaluation sets")
    if {normalize(x) for x in training} & {normalize(x) for x in evaluation}:
        raise ValueError("Training/evaluation overlap")
    return training, evaluation


def byte_budget_split(uk_lines, en_lines, total_cap, share=75, seed=42):
    if not (1 <= share <= 99) or total_cap < 2048:
        raise ValueError("Invalid training byte budget or ratio")
    max_total = min(total_cap, line_bytes(uk_lines)*100//share,
                    line_bytes(en_lines)*100//(100-share))
    uk_target = max_total*share//100
    en_target = max_total-uk_target
    return (select_to_budget(uk_lines, uk_target, seed+1),
            select_to_budget(en_lines, en_target, seed+2))


def train_cc0_tokenizer(uk_lines, en_lines, folder: Path, vocab=1024):
    """The resulting .model stays in runs/ and is intentionally not committed."""
    import sentencepiece as spm
    if vocab < 384:
        raise ValueError("Vocabulary too small for byte fallback")
    folder.mkdir(parents=True, exist_ok=True)
    ua, en = byte_budget_split(uk_lines, en_lines, 25000, share=75)
    mixed = ua + en
    random.Random(42).shuffle(mixed)
    trainpath = folder / "cc0_train.txt"
    write_lines(trainpath, mixed)
    spm.SentencePieceTrainer.train(
        input=str(trainpath), model_prefix=str(folder / "uk75_bpe"),
        vocab_size=vocab, model_type="bpe", byte_fallback=True,
        character_coverage=1.0, normalization_rule_name="identity",
        remove_extra_whitespaces=False, add_dummy_prefix=False,
        hard_vocab_limit=False, shuffle_input_sentence=False,
        bos_id=1, eos_id=2, pad_id=-1, unk_id=0, num_threads=2,
    )
    sp = spm.SentencePieceProcessor(model_file=str(folder / "uk75_bpe.model"))
    if sp.vocab_size() != vocab or sp.bos_id() < 0 or sp.eos_id() < 0:
        raise ValueError("Invalid trained SentencePiece vocabulary/special token IDs")
    for line in uk_lines[:10]+en_lines[:10]:
        if sp.decode(sp.encode(line)) != line:
            raise ValueError("Tokenizer is not lossless")
    return sp, {"uk_training_bytes": line_bytes(ua),
                "en_training_bytes": line_bytes(en), "vocab": sp.vocab_size(),
                "training_samples": len(mixed)}


def init_piece_vectors(sp, old_tokenizer, old_embedding, unk_id):
    """Mean source-model token embeddings for each new *decoded* token piece."""
    import torch
    initial = torch.empty((sp.vocab_size(), old_embedding.shape[1]),
                          device=old_embedding.device, dtype=old_embedding.dtype)
    fallback = old_embedding.mean(dim=0)
    bos = old_tokenizer.bos_token_id
    eos = old_tokenizer.eos_token_id
    for i in range(sp.vocab_size()):
        if i == sp.bos_id() and bos is not None:
            initial[i] = old_embedding[bos]
        elif i == sp.eos_id() and eos is not None:
            initial[i] = old_embedding[eos]
        elif i == sp.unk_id() and unk_id is not None:
            initial[i] = old_embedding[unk_id]
        elif sp.id_to_piece(i).startswith("<0x"):
            # Single raw byte often cannot be decoded to a Unicode character.
            initial[i] = fallback
        else:
            decoded = sp.decode([i])
            ids = old_tokenizer.encode(decoded, add_special_tokens=False)
            if ids and "\ufffd" not in decoded:
                initial[i] = old_embedding[ids].mean(dim=0)
            else:
                initial[i] = fallback
    return initial


def replace_embeddings(model, sp, original_tokenizer):
    """Replace both input and output LM-head embeddings, keeping other weights."""
    import torch
    with torch.no_grad():
        original = model.get_input_embeddings().weight.detach().clone()
        old_unk = original_tokenizer.unk_token_id
        vectors = init_piece_vectors(sp, original_tokenizer, original, old_unk)
        output = model.get_output_embeddings().weight
        new_output_vectors = (vectors if output.data_ptr() == model.get_input_embeddings().weight.data_ptr()
                              else init_piece_vectors(sp, original_tokenizer, output.detach().clone(), old_unk))
        model.resize_token_embeddings(sp.vocab_size(), mean_resizing=False)
        model.get_input_embeddings().weight.copy_(vectors)
        model.get_output_embeddings().weight.copy_(new_output_vectors)
        model.config.bos_token_id = sp.bos_id()
        model.config.eos_token_id = sp.eos_id()
        if getattr(model, "generation_config", None):
            model.generation_config.bos_token_id = sp.bos_id()
            model.generation_config.eos_token_id = sp.eos_id()
    return model


def ids_source(tokenizer, text):
    seq = tokenizer.encode(text, add_special_tokens=False)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if bos is None or eos is None:
        raise ValueError("Pretrained model must specify BOS and EOS")
    return [bos]+seq+[eos]


def ids_new(sp, text):
    return [sp.bos_id()] + sp.encode(text) + [sp.eos_id()]


def choose_paired_texts(texts, original_encode, candidate_encode, sample_size, max_text_tokens=80):
    """Evaluate identical, pre-filtered sentences under both tokenizers."""
    if sample_size <= 0:
        raise ValueError("At least one paired sample is needed")
    selection = []
    for text in texts:
        if (3 <= len(original_encode(text)) <= max_text_tokens+2 and
                3 <= len(candidate_encode(text)) <= max_text_tokens+2):
            selection.append(text)
            if len(selection) == sample_size:
                break
    if len(selection) < sample_size:
        raise ValueError(f"Not enough paired sentences: {len(selection)}/{sample_size}")
    return selection


def score_sentences(model, texts, encoding, max_text_tokens=80):
    """Causal likelihood per UTF-8 byte, excluding BOS and including EOS."""
    import torch
    import torch.nn.functional as F
    total_nll, total_bytes, token_count, skipped, elapsed = 0., 0, 0, 0, 0.
    model.eval()
    for text in texts:
        ids = encoding(text)
        if len(ids) > max_text_tokens+2 or len(ids) < 3:
            skipped += 1
            continue
        values = torch.tensor([ids], dtype=torch.long)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=values, use_cache=False).logits[:, :-1, :]
            labels = values[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   labels.reshape(-1), reduction="sum")
        elapsed += time.perf_counter()-t0
        total_nll += float(loss)
        total_bytes += len(text.encode("utf-8"))
        token_count += len(ids)-2
    if not total_bytes:
        raise ValueError("No texts within the context limit")
    return {"bits_per_utf8_byte": round(total_nll/(total_bytes*math.log(2)), 4),
            "text_tokens": token_count, "texts_evaluated": len(texts)-skipped,
            "texts_skipped_by_length": skipped,
            "bytes": total_bytes, "forward_seconds_cpu": round(elapsed, 3)}


def tune_only_embeddings(model, sp, examples, steps=4, lr=0.0003):
    import torch
    import torch.nn.functional as F
    if not (0 <= steps <= 40) or lr <= 0:
        raise ValueError("Unsafe training step count or learning rate")
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    model.get_output_embeddings().weight.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    model.train()
    trained = 0
    for text in examples:
        if trained >= steps:
            break
        ids = ids_new(sp, text)
        if len(ids) < 3 or len(ids) > 66:
            continue
        tensor = torch.tensor([ids], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=tensor, use_cache=False).logits[:, :-1, :]
        labels = tensor[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1))
        loss.backward()
        optimizer.step()
        trained += 1
    model.eval()
    return trained


def experiment(model_id=DEFAULT_MODEL, vocab=1024, sample_uk=10, sample_en=10,
               tuning_steps=4):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(42)
    torch.set_num_threads(2)
    corpora, metadata = {}, {}
    for lang, limit in (("uk", 3000), ("en", 2000)):
        corpus, provenance = load_cc0(download(SOURCES[lang]), lang, set(), limit)
        train, test = disjoint_cc0(corpus)
        corpora[lang] = (train, test)
        metadata[lang] = {
            "source_url": SOURCES[lang],
            "source_sha256": provenance["compressed_sha256"],
            "training_samples": len(train), "heldout_samples": len(test),
            "license": "CC0 export", "split": "stable_sha256_by_sentence",
        }
    train_uk, held_uk = corpora["uk"]
    train_en, held_en = corpora["en"]
    output = Path("runs/llm_adapter_v06")
    sp, training = train_cc0_tokenizer(train_uk, train_en, output, vocab)
    base_tok = AutoTokenizer.from_pretrained(model_id)
    # Pairwise eligibility is decided BEFORE any model forward pass, on CC0 test.
    # Using the same original bytes avoids misleading cross-tokenizer comparisons.
    heldout = {
        "uk": choose_paired_texts(
            held_uk, lambda t: ids_source(base_tok, t), lambda t: ids_new(sp, t), sample_uk),
        "en": choose_paired_texts(
            held_en, lambda t: ids_source(base_tok, t), lambda t: ids_new(sp, t), sample_en),
    }
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    base_model.to("cpu").eval()
    # Base model scores only the same CC0 texts, never new weights with old IDs.
    baseline = {lang: score_sentences(base_model, texts, lambda t: ids_source(base_tok,t))
                for lang,texts in heldout.items()}
    model_sha = getattr(base_model.config, "_commit_hash", None)
    sample_checksums = {
        lang: hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
        for lang, texts in heldout.items()
    }
    replace_embeddings(base_model, sp, base_tok)
    transplant = {lang: score_sentences(base_model, texts, lambda t: ids_new(sp,t))
                  for lang,texts in heldout.items()}
    # For this engineering pilot, training uses CC0 TRAIN only.
    trained_steps = tune_only_embeddings(base_model, sp, train_uk, tuning_steps)
    adapted = {lang: score_sentences(base_model, texts, lambda t: ids_new(sp,t))
               for lang,texts in heldout.items()}
    for lang in ("uk", "en"):
        same_count = {part[lang]["texts_evaluated"] for part in (baseline, transplant, adapted)}
        same_bytes = {part[lang]["bytes"] for part in (baseline, transplant, adapted)}
        if same_count != {len(heldout[lang])} or len(same_bytes) != 1:
            raise ValueError(f"Unpaired evaluation data detected for {lang}")
    return {
        "experiment": "tiny_open_llm_embedding_transplant_engineering_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pretrained_model": model_id, "model_revision": model_sha,
        "model_license": "Apache-2.0 (see model card)",
        "corpora": metadata, "tokenizer_training": training,
        "paired_evaluation": {"same_sentences_all_conditions": True,
                              "sha256_by_language": sample_checksums,
                              "text_count": {lang: len(texts) for lang, texts in heldout.items()}},
        "adapter_training_steps": trained_steps, "trainable": "input/output embeddings only",
        "measure": "nll_bits_per_original_utf8_byte",
        "results": {"original_tokenizer": baseline, "new_vocab_unadapted": transplant,
                    "new_vocab_after_training": adapted},
        "limitations": [
            "Engineering integration smoke test, NOT an unbiased held-out language quality benchmark.",
            "The same Tatoeba source was used for previous v0.5 tokenizer evaluation; results must not be promoted as new unseen evidence.",
            "All baseline/candidate numbers per language are from the exact same hashed paired sentence subset.",
            "CC0 Ukrainian corpus is tiny; tokenizer trained only on split CC0 data.",
            "Embedding transplant and a few tiny training steps cannot establish language-model quality or competitive downstream performance.",
            "Bits per original UTF-8 byte (including EOS) is comparable across distinct tokenizers; tokens-per-token/perplexity is NOT.",
            "Raw CPU forward time on shared GitHub runner is illustrative, not a controlled production latency measurement.",
            "Neither the model weights nor the tokenizer are published; model licensing/privacy due diligence required.",
            "Not GPT-6 or ChatGPT token/cost savings.",
        ]}


def markdown(report):
    lines = ["# UTP v0.6: real open-weight language model integration", "",
             "**Engineering demonstration, not improved Ukrainian LLM quality.**", "",
             f"Backbone: {report['pretrained_model']}; revision: {report['model_revision']}.",
             f"New tokenizer vocabulary: {report['tokenizer_training']['vocab']}.",
             f"Training steps on embeddings only: {report['adapter_training_steps']}.", "",
             "| Condition | Ukrainian bits/UTF-8 byte | English bits/UTF-8 byte |",
             "|---|---:|---:|"]
    for label, result in report["results"].items():
        lines.append(f"| {label} | {result['uk']['bits_per_utf8_byte']} | {result['en']['bits_per_utf8_byte']} |")
    lines += ["", "Lower bits per byte is better for language-model likelihood; this test does not evaluate task performance.", "", "## Caveats", ""]
    lines += ["- "+s for s in report["limitations"]]
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--sample-uk", type=int, default=10)
    p.add_argument("--sample-en", type=int, default=10)
    p.add_argument("--tuning-steps", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/llm_adapter_v06.json"))
    args=p.parse_args()
    report=experiment(args.model, args.vocab, args.sample_uk, args.sample_en, args.tuning_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__=="__main__":
    main()
