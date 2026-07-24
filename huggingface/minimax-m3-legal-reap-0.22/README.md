---
license: other
license_name: minimax-community
license_link: LICENSE
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- minimax
- minimax-m3
- legal-ai
- legal-reasoning
- mixture-of-experts
- moe
- reap
- expert-pruning
- text-generation
base_model: MiniMaxAI/MiniMax-M3
---

# MiniMax-M3 Legal REAP 0.22

MiniMax-M3 Legal REAP 0.22 is a PLI Labs **research artifact**: a text-only,
expert-pruned derivative of MiniMax-M3, produced to test whether a very large
sparse mixture-of-experts model can be compressed around legal work (drafting,
source-grounded synthesis, adversarial review, abstention, and long-context
document reasoning) rather than around coding benchmarks.

**This is an early, unhealed research checkpoint.** It has not been recovery-tuned,
it carries no published evaluation numbers, and it is held under review. It is not
a legal assistant, and its outputs may be degraded relative to the base model. All
legal use requires attorney supervision and independent source verification.

Built with MiniMax M3.

## Release Status

| | |
| --- | --- |
| Status | Research artifact — **unhealed raw prune, under review** |
| Evaluation | **None published.** No aggregate benchmark tables exist yet |
| Healing | Recovery fine-tune (heal) is the next pipeline stage — **not yet applied** |
| Verification | Held-out KL / faithfulness verification is **pending** |
| Recommended use | Research and internal evaluation only |

The public claim is deliberately narrow: this is a structurally complete,
legal-calibrated expert-prune of MiniMax-M3, released as a research artifact so the
method and the checkpoint can be inspected. It is **not** a benchmark-backed release
and should not be treated as one until healing and verification complete and
aggregate results are published.

## Model Details

| Field | Value |
| --- | --- |
| Model family | MiniMax-M3 (native multimodal MoE; this derivative is **text-only**) |
| Publisher | PLI Labs / Proprietary Legal Intelligence |
| Base model | `MiniMaxAI/MiniMax-M3` |
| Pruning method | REAP-style routed-expert pruning (observation-driven saliency) |
| Prune ratio | 0.22 (floor-enforced, per-layer adaptive) |
| Working precision | bf16 |
| Format | Transformers / safetensors checkpoint (original split-expert layout) |
| Total parameters | ~330B after prune (base ~428B) |
| Active parameters | ~23B (top-4 routing preserved — active count essentially unchanged) |
| Intended domain | Lawyer-supervised legal drafting, review, and analysis |
| Contact | nick@proprietarylegal.com |

### What changed structurally

The base MiniMax-M3 text backbone has 60 decoder layers: layers 0–2 are dense
(no routed experts) and layers 3–59 are MoE, with 128 routed experts plus 1
always-on shared expert per MoE layer and top-4 sigmoid routing.

This derivative:

- **prunes only the routed experts** on the 57 MoE layers, using a per-layer
  adaptive schedule at an overall 0.22 ratio. Retained routed-expert counts vary by
  layer between **89 and 111 of 128** (deeper, more separable layers give up more
  experts; capability-carrying layers are protected by floors);
- **leaves layers 0–2 (dense) untouched**;
- **preserves the shared expert** on every MoE layer;
- **preserves top-4 routing**, so the per-token active-parameter budget is
  essentially unchanged; the reduction is in total stored parameters;
- **drops the vision tower and multimodal projector** — this is a text-only
  checkpoint of the `MiniMaxM3SparseForCausalLM` text stack. MiniMax Sparse
  Attention (MSA), the learned router, routing bias, layernorms, embeddings, and
  the MTP modules are copied unchanged. The router weight and routing-bias rows for
  dropped experts are removed by row-slicing (no numeric renormalization).

## Intended Use

This artifact is intended for:

- research on legal-domain MoE compression;
- private, local evaluation by legal teams and model researchers;
- lawyer-supervised legal drafting and revision assistance **in an evaluation
  setting**;
- long-form legal and factual synthesis, source-grounded summarization, and
  retrieval-sensitive analysis under review.

The intended deployment pattern is local or on-premises evaluation on hardware
controlled by the lawyer, firm, or legal organization.

## Out-of-Scope Use

Do not use this artifact for:

- unsupervised legal advice;
- final court filings without attorney review;
- citation generation without independent verification;
- client-specific work where the user cannot verify the source record;
- any production deployment (this is an unhealed, unevaluated research checkpoint);
- non-legal capability claims (coding, general chat, multimodal — the vision path
  is removed and no capability is claimed).

## Method Summary

The prune direction is legal, not code. Routed-expert saliency was measured by
running a **private legal-domain calibration corpus** through an observer that
records, per expert, the REAP saliency signal (router gate value combined with
activation magnitude over the active-token set). The calibration corpus spans legal
drafting, summarization, adversarial review, abstention, and long-context
trial-transcript roles (975 rows across 13 length-bucketed components); its
contents are private and not distributed. Per-role observations were combined with
a role-weighted merge that emphasizes legal drafting.

A per-layer **adaptive** schedule then set each layer's retained-expert count under
capability floors (protecting under-covered layers and low-confidence carriers),
and the lowest-saliency experts were dropped per layer. The checkpoint was written
by a **streaming split-to-split serializer** that rewrites the on-disk shards
directly and never loads the full model, touching roughly one shard buffer plus one
tensor of peak memory. See the accompanying GitHub documentation for method detail.

The saliency criterion follows REAP (Lasby et al., "REAP the Experts: Why Pruning
Prevails for One-Shot MoE Compression", arXiv:2510.13999).

## Evaluation Status

**No evaluation numbers are published.** This checkpoint is the raw, unhealed prune.
Field evidence and internal research indicate that MiniMax-M3's experts are
near-orthogonal (little merge headroom) and that the honest cold-drop safe band for
legal work is roughly 0.16–0.25, with the losses concentrated in knowledge,
faithfulness, refusal, and long-context behavior rather than surface fluency —
axes that coding-style benchmarks do not see. A recovery heal and held-out
KL / faithfulness verification are the required next gates before any quality claim.

## Provenance

| Item | Value |
| --- | --- |
| Base model | `MiniMaxAI/MiniMax-M3` |
| Base revision | `MiniMaxAI/MiniMax-M3` (local bf16 conversion of the official release; no upstream revision pin recorded) |
| Prune ratio | 0.22 (floor-enforced, per-layer adaptive) |
| Selection metric | REAP saliency (canonical `reap` metric) |
| Seed | 42 |
| Cut-plan hash | `2e71105f68e30c81073623dc613448c24e94c06356d82414b9db86a78dd020c8` |
| Retained experts/layer | 89–111 of 128 (per-layer adaptive) |
| Dense layers preserved | 0, 1, 2 |
| Shard total size | 670,639,198,956 bytes (~624.6 GiB, 126 shards) |
| Transformers baseline | 4.52.4 (base config); load path fuses split experts at load |

A machine-readable provenance sidecar (`reap_prune_provenance.json`) and an
expert-count census accompany the checkpoint and record the base index hash, the
observation hash, the full per-layer cut plan, and the cut-plan hash.

## Privacy Boundary

This release does not include client names, client files, closed-client materials,
attorney-client communications, privileged work product, private prompts,
calibration row contents, internal hostnames or IP addresses, local file paths,
credentials, or tokens. The calibration corpus is described only generically.

## License Boundary

This is a derivative work of MiniMax-M3 and remains subject to the **MiniMax
Community License** (`license_name: minimax-community`); the upstream `LICENSE` is
redistributed with this checkpoint. In summary, and without replacing the license
text:

- The license permits use, modification, and redistribution of the software and
  derivative works, including publishing this pruned checkpoint.
- **Commercial use** carries additional obligations: you must prominently display
  **"Built with MiniMax M3"**, and you must send a notice to MiniMax (or, above a
  revenue threshold, obtain prior written authorization) per the license terms.
- The license includes a prohibited-uses appendix (no unlawful content, no military
  use, no exploitation of minors, no harmful disinformation, no discriminatory
  content).

Review the upstream license in full before use, especially for any commercial
deployment.

## Citation

> PLI Labs. MiniMax-M3 Legal REAP 0.22 model card. Proprietary Legal Intelligence,
> 2026. Derived from MiniMaxAI/MiniMax-M3. Pruning method after Lasby et al.,
> arXiv:2510.13999.

## Links

- PLI Labs: https://proprietarylegal.com
- Research site: https://proprietarylegal.ai
- GitHub documentation: https://github.com/ProprietaryLegal/pli-labs
- GitHub model card: https://github.com/ProprietaryLegal/pli-labs/blob/main/reaps/minimax-m3-legal-reap-0.22/MODEL_CARD.md
- REAP method: https://arxiv.org/abs/2510.13999
- MiniMax-M3 base: https://huggingface.co/MiniMaxAI/MiniMax-M3
