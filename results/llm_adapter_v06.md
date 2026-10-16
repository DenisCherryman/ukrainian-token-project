# UTP v0.6: real open-weight language model integration

**Engineering demonstration, not improved Ukrainian LLM quality.**

Backbone: HuggingFaceTB/SmolLM2-135M; revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2.
New tokenizer vocabulary: 1024.
Training steps on embeddings only: 4.

| Condition | Ukrainian bits/UTF-8 byte | English bits/UTF-8 byte |
|---|---:|---:|
| original_tokenizer | 1.9755 | 1.2429 |
| new_vocab_unadapted | 2.5298 | 4.6477 |
| new_vocab_after_training | 2.4847 | 4.6349 |

Lower bits per byte is better for language-model likelihood; this test does not evaluate task performance.

## Caveats

- Engineering integration smoke test, NOT an unbiased held-out language quality benchmark.
- The same Tatoeba source was used for previous v0.5 tokenizer evaluation; results must not be promoted as new unseen evidence.
- CC0 Ukrainian corpus is tiny; tokenizer trained only on split CC0 data.
- Embedding transplant and a few tiny training steps cannot establish language-model quality or competitive downstream performance.
- Bits per original UTF-8 byte (including EOS) is comparable across distinct tokenizers; tokens-per-token/perplexity is NOT.
- Raw CPU forward time on shared GitHub runner is illustrative, not a controlled production latency measurement.
- Neither the model weights nor the tokenizer are published; model licensing/privacy due diligence required.
- Not GPT-6 or ChatGPT token/cost savings.
