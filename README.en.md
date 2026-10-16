# Ukrainian Token Project — English summary

UTP is an open-source research toolkit for measuring and investigating Ukrainian-language tokenization efficiency. It does **not** modify ChatGPT or any deployed language model.

Version 0.3 implements reproducible document-disjoint train/dev/test preparation for Universal Dependencies datasets, exact sentence decontamination, SentencePiece mixture training, development-only model selection, held-out evaluation, paired **document-cluster bootstrap**, and aligned Ukrainian/English token-count comparisons using public `tiktoken` encodings.

**Research status:** source code and synthetic end-to-end smoke tests were verified, but real-corpus experiments have not been executed in the build environment. No production savings or downstream LLM quality improvements have been established.

For setup and licensing details, see the Ukrainian [README](README.md) and [methodology](docs/METHODOLOGY.md). Ukrainian UD-IU is **CC BY-NC-SA 4.0**; English EWT includes third-party text copyrights. These corpora are **not redistributed** by UTP. MIT applies only to project-authored code.

Contributions welcome: document-disjoint benchmarks, licensed parallel text pairs, replicable negative results, and rigorous tokenizer-to-LLM adaptation studies.
