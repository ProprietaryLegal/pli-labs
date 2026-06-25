# Stack And Settings: Qwen3.5 Legal REAP

This file is the public-safe settings summary for the Qwen3.5 legal REAP
release. It preserves the technical facts needed to understand and reproduce
the release direction without exposing private source paths, client materials,
or host-specific runbooks.

## Base Model

| Field | Value |
| --- | --- |
| Base model | Qwen/Qwen3.5-122B-A10B |
| Base license | Apache-2.0 |
| Model type | qwen3_5_moe_text |
| Architecture | Qwen3_5MoeForCausalLM |
| Hidden size | 3072 |
| Original layers | 48 |
| Original routed experts | 256 per MoE layer |
| Experts per token | 8 routed experts |
| Attention heads | 32 |
| Key/value heads | 2 |
| Head dimension | 256 |
| Vocabulary size | 248,320 |

## Calibration Direction

The calibration mix was legal-work oriented and included approximately 1,409
calibration records across these public-safe categories:

- long-form legal and factual synthesis;
- agentic legal-workflow prompts;
- structured JSON/tool-output prompts;
- refusal and escalation prompts;
- long-context prompts;
- authored legal reasoning and correction prompts.

The public release does not include the private calibration records or identify
the legal source items used to build them.

## REAP 0.16 Settings

| Setting | Value |
| --- | --- |
| Compression ratio setting | 0.16 |
| Prune method | REAP-style routed-expert pruning |
| Cluster method | agglomerative |
| Expert similarity | TTM |
| Linkage method | average |
| Frequency penalty | enabled |
| Router renormalization | enabled |
| Preserve outlier experts | disabled for this run |
| Preserve super experts | disabled for this run |
| Seed | 42 |
| Working precision | bfloat16 |
| Observer batch size | 1 |
| Batches per category | 1024 |
| Observer max length | 8192 tokens |
| Distance measure | angular |
| Shuffle calibration split | enabled |
| Chunked pruning metrics | enabled |

The resulting checkpoint keeps 216 routed experts per MoE layer. Because the
base model has 48 MoE layers, the release removes 1,920 routed experts in
total while preserving the top-8 routing pattern.

## Legal REAP 0.16 Artifact Facts

| Field | Value |
| --- | --- |
| HF repo | ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap |
| Layers | 48 |
| Routed experts per MoE layer | 216 |
| Experts per token | 8 |
| Indexed weight entries | 31,839 |
| Indexed tensor bytes | 207,972,470,784 |
| Estimated total parameters | about 104B |
| Local shard count | 5 safetensors shards |
| Local artifact size | about 194G |

File inventory:

| File | Bytes |
| --- | ---: |
| model-00001-of-00005.safetensors | 47,290,109,968 |
| model-00002-of-00005.safetensors | 49,872,423,688 |
| model-00003-of-00005.safetensors | 48,316,170,528 |
| model-00004-of-00005.safetensors | 49,872,423,648 |
| model-00005-of-00005.safetensors | 12,625,862,688 |
| model.safetensors.index.json | 3,359,363 |
| tokenizer.json | 19,989,426 |
| tokenizer_config.json | 1,123 |
| chat_template.jinja | 7,756 |
| config.json | 2,531 |
| generation_config.json | 219 |

## Layerdrop6 Settings

The Layerdrop6 derivative starts from the 0.16 REAP checkpoint and removes six
decoder layers:

```text
8, 9, 12, 13, 16, 17
```

This reduces the model from 48 layers to 42 layers while keeping the REAP-0.16
expert count of 216 routed experts per remaining MoE layer.

## Layerdrop6 Artifact Facts

| Field | Value |
| --- | --- |
| HF repo | ProprietaryLegal/Qwen3.5-91b-a10b-LegalReap-Layerdrop6 |
| Parent checkpoint | ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap |
| Layers | 42 |
| Routed experts per MoE layer | 216 |
| Experts per token | 8 |
| Indexed weight entries | 27,855 |
| Indexed tensor bytes | 182,327,694,336 |
| Estimated total parameters | about 91B |
| Local shard count | 4 safetensors shards |
| Local artifact size | about 170G |

File inventory:

| File | Bytes |
| --- | ---: |
| model-00001-of-00004.safetensors | 49,995,940,344 |
| model-00002-of-00004.safetensors | 48,486,060,768 |
| model-00003-of-00004.safetensors | 49,872,423,688 |
| model-00004-of-00004.safetensors | 33,977,224,392 |
| model.safetensors.index.json | 2,938,189 |
| tokenizer.json | 19,989,426 |
| tokenizer_config.json | 1,123 |
| chat_template.jinja | 7,756 |
| config.json | 2,387 |
| generation_config.json | 219 |

## What Was Not Released

No separate expert-merged Qwen3.5 checkpoint is included in this publication.
Expert merging remains a follow-on research lane. The public release consists
of:

1. Qwen3.5-104b-a10b-LegalReap, the 48-layer expert-reduced checkpoint; and
2. Qwen3.5-91b-a10b-LegalReap-Layerdrop6, the 42-layer derivative.

Raw private run configuration files that contain local filesystem paths are
not published to Hugging Face. This file is the public-safe replacement.
