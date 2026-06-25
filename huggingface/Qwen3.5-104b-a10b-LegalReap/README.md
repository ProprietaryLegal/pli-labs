---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- qwen
- qwen3.5
- legal-ai
- legal-reasoning
- mixture-of-experts
- reap
- expert-pruning
- text-generation
base_model: Qwen/Qwen3.5-122B-A10B
---

# Qwen3.5-104b-a10b-LegalReap

Qwen3.5-104b-a10b-LegalReap is PLI Labs' conservative legal-domain
expert-reduced Qwen3.5 candidate. It starts from Qwen3.5-122B-A10B and removes
the stable low-saliency routed-expert core observed under a legal calibration
workload, reducing each MoE layer from 256 routed experts to 216 while
preserving the original 48-layer depth, hidden size, tokenizer, and top-8
routing pattern.

This is a research/evaluation release, not an unsupervised legal advice system.
Legal use requires attorney supervision and independent source verification.

## Release Positioning

PLI Labs builds legal AI systems for lawyers who need private, inspectable, and
locally evaluable model workflows. This release applies that thesis to a large
open MoE model: instead of pruning around coding benchmarks, the calibration
direction was legal drafting, factual synthesis, source-grounded review,
workflow-sensitive reasoning, and conservative refusal behavior.

The public claim is deliberately specific:

- this is a legal-domain REAP candidate;
- it is expert-reduced, not expert-merged;
- the released files are suitable for research and controlled evaluation;
- Qwen3.5-122B-A10B appears prune-resistant, so the cut is intentionally
  conservative;
- public aggregate legal benchmark tables should be attached before broader
  validated-production claims are made.

## Model Details

| Field | Value |
| --- | --- |
| Model family | Qwen3.5 MoE |
| Publisher | PLI Labs / Proprietary Legal Intelligence |
| Release status | Research candidate |
| Base model | Qwen/Qwen3.5-122B-A10B |
| Base license | Apache-2.0 |
| Format | Transformers/safetensors checkpoint |
| Architecture | Qwen3_5MoeForCausalLM |
| Pruning method | REAP-style routed-expert pruning |
| Router handling | Renormalized over surviving routed experts |
| Hidden size | 3072 |
| Layers | 48 |
| Routed experts | 216 per MoE layer |
| Original routed experts | 256 per MoE layer |
| Experts per token | 8 routed experts |
| Attention heads | 32 |
| Key/value heads | 2 |
| Head dimension | 256 |
| Vocabulary size | 248,320 |
| Indexed weight entries | 31,839 |
| Indexed tensor bytes | 207,972,470,784 |
| Estimated total parameters | about 104B |
| Contact | nick@proprietarylegal.com |

## Intended Use

This candidate is intended for:

- lawyer-supervised legal drafting and revision assistance;
- long-form legal and factual synthesis;
- source-grounded summarization and review;
- retrieval-sensitive analysis;
- legal workflow/tool-use evaluation;
- conservative refusal or escalation when the record is incomplete;
- local or private evaluation by legal teams.

## Out-of-Scope Use

Do not use this candidate for:

- unsupervised legal advice;
- final court filings without attorney review;
- citation generation without independent verification;
- client-specific work where the user cannot verify the source record;
- production deployment based only on this model card;
- claims of general coding, chat, or benchmark superiority.

## Method Summary

The calibration direction was legal work, not generic chat or coding. Public
documentation intentionally does not identify private legal source items used
for calibration or evaluation.

The pruning procedure:

1. Observe routed-expert usage under a legal calibration workload.
2. Rank routed experts by REAP-style saliency.
3. Select the stable low-saliency core rather than forcing a larger prune.
4. Remove 40 routed experts per MoE layer, reducing 256 experts to 216.
5. Renormalize routing over survivors.
6. Save a complete Transformers/safetensors checkpoint.

The main technical finding is as important as the artifact: Qwen3.5-122B-A10B
appears prune-resistant. PLI Labs therefore treats 0.16 as the conservative
expert-reduction lane and uses quantization, evaluation, and possible future
expert-merge research as the next levers.

## Evaluation Status

Status: research candidate.

The checkpoint is structurally complete and suitable for evaluation. Public
aggregate legal benchmark tables are not yet attached to this card. Until those
tables are published, treat the model as a candidate for research and internal
evaluation rather than a validated production legal model.

## Privacy Boundary

This public release intentionally excludes:

- client names;
- matter names;
- legal source item titles;
- private calibration prompts;
- attorney-client communications;
- privileged work product;
- local hostnames, private paths, credentials, or tokens.

The raw run configuration included machine-local paths and is not uploaded to
this repository. A public-safe settings summary is maintained in the PLI Labs
GitHub documentation.

## Limitations

- Strong fluency is not proof of legal reliability.
- Expert pruning can preserve surface quality while degrading source fidelity
  or tool judgment.
- Public aggregate evaluation tables are still pending.
- This is not an expert-merged checkpoint; it is an expert-reduced REAP
  checkpoint.
- All legal outputs require attorney review and source verification.

## Links

- GitHub documentation: https://github.com/ProprietaryLegal/pli-labs
- Research packet: https://github.com/ProprietaryLegal/pli-labs/tree/main/research/qwen35-legal-reap
- Model card mirror: https://github.com/ProprietaryLegal/pli-labs/blob/main/reaps/Qwen3.5-104b-a10b-LegalReap/MODEL_CARD.md
- Proprietary Legal: https://proprietarylegal.com
- PLI research site: https://proprietarylegal.ai
