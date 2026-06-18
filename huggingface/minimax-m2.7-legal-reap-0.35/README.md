---
license: other
license_name: minimax-m2.7-license
license_link: https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- minimax
- minimax-m2.7
- legal-ai
- legal-reasoning
- mixture-of-experts
- reap
- expert-pruning
- text-generation
base_model: MiniMaxAI/MiniMax-M2.7
---

# MiniMax-M2.7 Legal REAP 0.35

MiniMax-M2.7 Legal REAP 0.35 is a PLI Labs benchmark-backed candidate for
legal-domain expert pruning. It was produced to test whether a large
mixture-of-experts model can be compressed around legal drafting, factual
synthesis, source-grounded analysis, refusal behavior, and workflow-sensitive
reasoning.

This is a research/evaluation release, not an unsupervised legal advice system.
All legal use requires attorney supervision and independent source verification.

Built with MiniMax M2.7.

## Release Positioning

PLI Labs builds legal AI systems for lawyers who need private, inspectable, and
locally evaluable model workflows. This candidate is part of that work: instead
of pruning around coding benchmarks, the calibration direction was legal work.

The public claim is deliberately narrow:

- this is a legal-domain REAP candidate;
- the released files are suitable for research and evaluation;
- internal evaluation is promising;
- public aggregate benchmark tables should be attached before broader release
  claims are made.

## Model Details

| Field | Value |
| --- | --- |
| Model family | MiniMax-M2.7 |
| Publisher | PLI Labs / Proprietary Legal Intelligence |
| Release status | Benchmark-backed candidate |
| Format | Transformers/safetensors checkpoint |
| Pruning method | REAP-style routed-expert pruning |
| Working precision | bf16 |
| Intended domain | Lawyer-supervised legal drafting, review, and analysis |
| Base model | MiniMaxAI/MiniMax-M2.7 |
| Contact | nick@proprietarylegal.com |

## Intended Use

This candidate is intended for:

- lawyer-supervised legal drafting and revision assistance;
- long-form legal and factual synthesis;
- source-grounded summarization;
- retrieval-sensitive analysis;
- conservative refusal or escalation when the record is incomplete;
- local or private evaluation by legal teams.

## Out-of-Scope Use

Do not use this candidate for:

- unsupervised legal advice;
- final court filings without attorney review;
- citation generation without independent verification;
- client-specific work where the user cannot verify the source record;
- production deployment based only on this model card.

## Evaluation Status

Status: benchmark-backed candidate.

PLI Labs internal legal evaluation indicates promising behavior on the target
comparison tasks. Public aggregate score tables are still pending. Until those
tables are published, treat this model as a candidate for research and internal
evaluation rather than a validated production legal model.

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
- GitHub model card: https://github.com/ProprietaryLegal/pli-labs/blob/main/reaps/minimax-m2.7-legal-reap-0.35/MODEL_CARD.md
