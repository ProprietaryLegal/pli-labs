---
license: other
license_name: minimax-m2.7-license
license_link: https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE
pipeline_tag: text-generation
language:
- en
tags:
- gguf
- llama.cpp
- minimax
- minimax-m2.7
- legal-ai
- mixture-of-experts
- reap
- expert-pruning
base_model: MiniMaxAI/MiniMax-M2.7
---

# MiniMax-M2.7 Legal REAP 0.35 GGUF

This repository contains the GGUF release for PLI Labs' MiniMax-M2.7 Legal REAP
0.35 candidate.

The file is intended for local serving experiments with GGUF-compatible
inference stacks. It is a research/evaluation artifact, not an unsupervised
legal advice system.

Built with MiniMax M2.7.

## Release Notes

- Format: GGUF.
- Candidate: MiniMax-M2.7 Legal REAP 0.35.
- Intended use: lawyer-supervised legal drafting, review, source-grounded
  analysis, and private local evaluation.
- Public status: benchmark-backed candidate; sanitized aggregate benchmark
  tables are pending.

## Public Positioning

PLI Labs publishes legal model work for lawyers and legal technologists who
need private, inspectable AI systems. The 0.35 GGUF is meant to make legal REAP
evaluation easier on local inference stacks without publishing private
deployment details.

## Privacy Boundary

This release does not include client files, closed-client materials,
attorney-client communications, privileged work product, private prompts,
internal host details, local file paths, credentials, or tokens.

## License Boundary

This model is a derivative of MiniMax-M2.7 and remains subject to the
MiniMax-M2.7 license. Review the upstream license before use, especially for
commercial use.

## Links

- PLI Labs: https://proprietarylegal.com
- Research site: https://proprietarylegal.ai
- GitHub documentation: https://github.com/ProprietaryLegal/pli-labs
- Source model card: https://github.com/ProprietaryLegal/pli-labs/blob/main/reaps/minimax-m2.7-legal-reap-0.35/MODEL_CARD.md
