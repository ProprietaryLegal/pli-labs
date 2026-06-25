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
- layerdrop
- expert-pruning
- text-generation
base_model: ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap
---

# Qwen3.5-91b-a10b-LegalReap-Layerdrop6

Qwen3.5-91b-a10b-LegalReap-Layerdrop6 is the depth-reduced companion to
PLI Labs' conservative Qwen3.5 legal REAP checkpoint. It starts from the
REAP-0.16 model and removes six low-impact decoder layers selected by a
calibration-driven layer-importance pass.

This release explores a second compression axis. REAP reduces routed expert
capacity; Layerdrop6 reduces depth. The result keeps 216 routed experts per
remaining MoE layer and reduces the network from 48 layers to 42 layers.

This is a research/evaluation release, not an unsupervised legal advice system.
Legal use requires attorney supervision and independent source verification.

## Release Positioning

The Layerdrop6 model is intentionally presented as a companion artifact, not as
a replacement for the 48-layer REAP-0.16 checkpoint. It is for evaluators who
want to compare legal behavior across:

- the original Qwen3.5-122B-A10B base model;
- the conservative expert-reduced REAP-0.16 checkpoint;
- a smaller depth-reduced derivative of that checkpoint.

## Model Details

| Field | Value |
| --- | --- |
| Model family | Qwen3.5 MoE |
| Publisher | PLI Labs / Proprietary Legal Intelligence |
| Release status | Research candidate |
| Parent checkpoint | ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap |
| Original base model | Qwen/Qwen3.5-122B-A10B |
| Base license | Apache-2.0 |
| Format | Transformers/safetensors checkpoint |
| Architecture | Qwen3_5MoeForCausalLM |
| Compression method | REAP expert reduction plus whole-layer drop |
| Dropped layer indices | 8, 9, 12, 13, 16, 17 |
| Hidden size | 3072 |
| Layers | 42 |
| Parent layers | 48 |
| Routed experts | 216 per remaining MoE layer |
| Original routed experts | 256 per MoE layer |
| Experts per token | 8 routed experts |
| Attention heads | 32 |
| Key/value heads | 2 |
| Head dimension | 256 |
| Vocabulary size | 248,320 |
| Indexed weight entries | 27,855 |
| Indexed tensor bytes | 182,327,694,336 |
| Estimated total parameters | about 91B |
| Contact | nick@proprietarylegal.com |

## Intended Use

This candidate is intended for:

- legal-domain compression research;
- side-by-side evaluation against the 48-layer REAP-0.16 model;
- lawyer-supervised legal drafting and synthesis experiments;
- source-grounded summarization and review experiments;
- local or private evaluation by legal teams.

## Out-of-Scope Use

Do not use this candidate for:

- unsupervised legal advice;
- final court filings without attorney review;
- citation generation without independent verification;
- production deployment based only on this model card;
- claims that layer removal improves legal capability without benchmark support.

## Method Summary

The parent REAP-0.16 checkpoint was produced by routed-expert pruning from
Qwen3.5-122B-A10B. This companion model then removes six decoder layers:

```text
8, 9, 12, 13, 16, 17
```

Those layer indices were selected from a calibration-driven layer-importance
pass. Public documentation intentionally does not identify private legal source
items used for calibration or evaluation.

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

## Limitations

- Layer removal can produce different degradation patterns from expert pruning.
- Strong fluency is not proof of legal reliability.
- Public aggregate evaluation tables are still pending.
- This is not an expert-merged checkpoint; it is a Layerdrop6 derivative of an
  expert-reduced REAP checkpoint.
- All legal outputs require attorney review and source verification.

## Links

- Parent model: https://huggingface.co/ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap
- GitHub documentation: https://github.com/ProprietaryLegal/pli-labs
- Research packet: https://github.com/ProprietaryLegal/pli-labs/tree/main/research/qwen35-legal-reap
- Model card mirror: https://github.com/ProprietaryLegal/pli-labs/blob/main/reaps/Qwen3.5-91b-a10b-LegalReap-Layerdrop6/MODEL_CARD.md
- Proprietary Legal: https://proprietarylegal.com
- PLI research site: https://proprietarylegal.ai
