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
- mixture-of-experts
- reap
- expert-pruning
- legal-ai
- b70
base_model: MiniMaxAI/MiniMax-M2.7
---

# MiniMax-M2.7 REAP 172B-A10B Q4_K_M GGUF

This repository contains a PLI Labs MiniMax-M2.7 REAP GGUF release intended for
local large-model serving research. It is the public B70-friendly GGUF artifact
from the PLI Labs owned-hardware research lane.

This is a research/evaluation artifact, not an unsupervised legal advice
system. All legal use requires attorney supervision and independent source
verification.

Built with MiniMax M2.7.

## Release Notes

- Format: GGUF.
- Quantization: Q4_K_M.
- Intended use: local research and lawyer-supervised legal AI evaluation.
- Hardware framing: tested as part of PLI Labs' B70-class local inference
  research.

The public card intentionally avoids exact launch flags, private topology,
service ports, internal file paths, and failure runbooks. Those details are
deployment-specific and are not needed to evaluate the public model artifact.

## Public Positioning

PLI Labs studies whether lawyers can run serious legal AI workloads on owned
hardware instead of sending sensitive work to opaque hosted systems. This GGUF
supports that research by providing a compact local-serving artifact for
large-model legal evaluation.

## Intended Use

- local serving experiments;
- legal drafting and review evaluation under lawyer supervision;
- source-grounded analysis tests;
- infrastructure comparison across GGUF-compatible runtimes.

## Out-of-Scope Use

- unsupervised legal advice;
- production deployment without legal and technical validation;
- final court filings without attorney review;
- claims that one hardware class or runtime is universally superior.

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
- B70 research documentation: https://github.com/ProprietaryLegal/pli-labs/tree/main/research/b70
- GitHub documentation: https://github.com/ProprietaryLegal/pli-labs
