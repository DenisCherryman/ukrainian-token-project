# UTP v0.6 — open-weight LLM tokenizer transplant (engineering only)

## Why this experiment exists

A tokenizer reducing input token count is not evidence that the underlying language model can use the new token IDs. Token IDs index learned input and output embeddings. Simply switching vocabulary without remapping/adapting the network invalidates language-model outputs.

This **engineering proof-of-concept** measures whether we can train a Ukrainian-weighted, lossless tokenizer on only CC0 Ukrainian/English text, remap its token vectors to an existing **Apache 2.0** 135M-parameter pretrained model, perform a handful of embedding-only training updates and score likelihood on separate CC0 sentences. This deliberately does **not** claim accuracy gains.

## Data boundaries

- Model: HuggingFaceTB/SmolLM2-135M, https://huggingface.co/HuggingFaceTB/SmolLM2-135M (Apache 2.0).
- Input: Tatoeba exports explicitly labeled **CC0** only; no UD text or UD-trained models in this workflow. Only download and process locally on the runner; do not commit raw sentences or model artifacts.
- Deterministic SHA-256 sentence-level 75/25 train/evaluation split in each language. Not author-disjoint; Tatoeba has limited Ukrainian CC0 examples.
- Train a BPE tokenizer of 1024 tokens with a 75/25 Ukrainian/English *UTF-8 byte budget*. Its size is much smaller than the original SmolLM2 vocabulary.
- Previous v0.5 research used the Tatoeba source already. **The evaluation portion must be described as an engineering check, not as a newly untouched research benchmark.**

## Embedding transplant

For each new SentencePiece piece, obtain the decoded string and tokenize it with the original pretrained tokenizer. Initialize the new input vector as the mean of those source token embeddings. Explicitly preserve pretrained BOS/EOS vectors where available. Invalid single-byte/SentencePiece special token pieces receive a fallback embedding. If the LM output head is untied, initialize that as well, then tie weights as defined by model config.

The rest of the transformer stays frozen. Optionally run up to four steps of embedding-only optimization on **CC0 TRAIN** examples. Do not publish the new tokenizer weights or finetuned model until the project has passed a separate license/privacy/quality review.

## Evaluation and expected outcome

On identical held-out sentences, record the aggregate negative log probability **in bits per ORIGINAL UTF-8 byte**, with EOS included. This normalization is required because a conventional per-token perplexity comparison is invalid when vocabulary and segmentation differ.

Conditions:
1. Unmodified pretrained model with its original tokenizer.
2. Same pretrained model after embedding surgery, before continuation.
3. Same model after a few tiny continuation steps in the new embedding tables only.

Report separately UA/EN token counts, bits-per-byte, CPU elapsed forward seconds, number of evaluated and skipped examples. **Lower bits-per-byte is better**. It is entirely possible (and expected with an almost untrained embedding mapping) that the adapted tokenizer reduces tokens yet heavily degrades language likelihood. This is a useful engineering finding, not a failure to be concealed.

CPU timings on the shared GitHub runner are illustrative, not comparable production latency measures.

## Repeat

On the [v0.6 workflow](../.github/workflows/llm_adapter.yml), GitHub Actions installs CPU PyTorch/Transformers, downloads the open model and CC0 exports, executes tests and stores only aggregate JSON/Markdown under results. A failure means do not report a measured LLM result.

The goal after this is **continued pretraining with a legal, large, high-quality corpus**, evaluations of task quality and speed under equivalent hardware, and a truly held-out Ukrainian benchmark from a new source. Tokenizer-only gains are not savings in GPT-6 or ChatGPT.
