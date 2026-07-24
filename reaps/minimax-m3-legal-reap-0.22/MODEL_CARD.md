# Model Card: MiniMax-M3 Legal REAP 0.22

## Summary

MiniMax-M3 Legal REAP 0.22 is a research artifact produced by PLI Labs to test
legal-domain expert pruning of a very large sparse mixture-of-experts model. The
prune direction is legal — preservation of legal drafting, long-form factual
synthesis, source-grounded analysis, abstention, and long-context document
reasoning — rather than a coding benchmark.

This is a **candidate research artifact, unhealed and under review**. The public
repository records the method, the known structural facts, intended use,
limitations, and status. No aggregate benchmark tables exist yet. Recovery healing
and held-out verification must complete before this card is treated as a
benchmark-backed or validated release.

## Model Details

| Field | Value |
| --- | --- |
| Model family | MiniMax-M3 (native multimodal MoE; this derivative is text-only) |
| Release name | MiniMax-M3 Legal REAP 0.22 |
| Release status | Research artifact — unhealed raw prune, under review |
| Publisher | PLI Labs / Proprietary Legal Intelligence |
| Base model | `MiniMaxAI/MiniMax-M3` |
| Pruning method | REAP-style routed-expert pruning (observation-driven saliency) |
| Base text backbone | 60 decoder layers (0–2 dense, 3–59 MoE); 128 routed + 1 shared expert per MoE layer; top-4 sigmoid routing; MiniMax Sparse Attention; MTP=7 |
| Prune ratio | 0.22 (floor-enforced, per-layer adaptive) |
| Retained routed experts | 89–111 of 128 per MoE layer (per-layer adaptive) |
| Dense layers | 0, 1, 2 preserved (never pruned) |
| Shared expert | preserved on every MoE layer |
| Working precision | bf16 |
| Total parameters | ~330B after prune (base ~428B) |
| Active parameters | ~23B (top-4 routing preserved) |
| Public weights | Distributed through Hugging Face |
| Public docs | this model card, the release README, and the method documents |
| Contact | nick@proprietarylegal.com |

## Intended Use

This artifact is intended for research and evaluation of legal-domain model
compression. The target workflows are:

- legal drafting and revision assistance under lawyer supervision;
- long-form legal and factual synthesis;
- source-grounded summarization;
- retrieval-sensitive analysis;
- conservative refusal / escalation when the record is incomplete.

The intended deployment pattern is local or on-premises evaluation on hardware
controlled by the lawyer, firm, or legal organization.

## Out-of-Scope Use

This artifact should not be used for:

- unsupervised legal advice;
- final court filings without attorney review;
- citation generation without independent source verification;
- client-specific work where the user cannot verify source support;
- any production deployment (it is unhealed and unevaluated);
- non-legal capability claims (coding, general chat) or multimodal use (the vision
  path is removed).

## Calibration Signal

The calibration direction was legal. Routed-expert saliency was observed by running
a private legal-domain calibration corpus through an observer that records, per
expert, the REAP saliency signal (router gate value combined with activation
magnitude over the active-token set), so the prune preserves the experts legal work
actually uses.

Public-safe calibration facts:

- calibration direction: legal drafting, summarization, adversarial review,
  abstention, and long-context trial-transcript roles;
- 975 rows across 13 length-bucketed components (context tiers up to 16,384 tokens);
- per-role observations combined with a role-weighted merge emphasizing legal
  drafting;
- optimization target: preserve legal capability, not coding-benchmark performance.

The public repository does not include private prompts, client matter details,
privileged work product, private file paths, calibration row contents, or raw
client-source examples.

## Pruning Procedure

The run used REAP-style routed-expert pruning, adapted for MiniMax-M3's top-4
sparse-attention architecture and executed model-free:

1. Observe the legal calibration workload, recording per-expert saliency on the 57
   MoE layers (layers 0–2 are dense and carry no routed experts).
2. Merge per-role observations with a role-weighted recombination emphasizing legal
   drafting.
3. Compute a per-layer **adaptive** retained-expert schedule at overall ratio 0.22,
   under capability floors that protect under-covered layers and low-confidence
   carriers.
4. For each MoE layer, drop the lowest-saliency experts down to the layer's
   retained count (89–111 of 128).
5. Serialize a smaller checkpoint by a **streaming split-to-split** rewrite that
   never loads the model: retained experts are copied byte-identical and renumbered
   into dense positions; router and routing-bias rows are row-sliced to the retained
   set; the vision tower / projector are dropped; everything else is copied
   unchanged.
6. Emit a provenance sidecar and an expert-count census alongside the shards.

Structural facts recorded at serialization time (fill at publish):

- `<FILL: shard count>` safetensors shards + index;
- no missing indexed shard keys;
- config showing the per-layer retained-expert counts;
- cut-plan hash `<FILL: cut_plan_hash>`.

## Evaluation Status

**Current status: unhealed research artifact — no published evaluation.**

This is the raw prune, before any recovery tuning. Field evidence and internal
research indicate MiniMax-M3's experts are near-orthogonal (little merge headroom),
that the honest cold-drop safe band for legal work is roughly 0.16–0.25, and that
the losses of deep MoE pruning concentrate in knowledge, faithfulness, refusal, and
long-context behavior — axes coding benchmarks do not measure. The next public
validation update should include:

- a recovery-heal result;
- held-out KL / faithfulness and top-1 flip-rate against the base model;
- aggregate legal task-family scores with at least two seeds and verified served
  identity;
- comparison against the base model;
- known failures, regressions, and the recommended deployment envelope.

## Hardware and Serving Notes

The prune was produced offline. Because MiniMax Sparse Attention has no current
Ampere/Volta kernel and the model's serving stack targets newer accelerators,
serving this checkpoint today generally means a text-only dense-fallback path
(e.g. a GGUF/llama.cpp route as it matures). The value of the artifact is durable,
portable, pruned weights: prune now, let backends mature. Deployment should be
validated on the intended runtime, precision, and context tier before any use.

Public hardware discussion is limited to hardware **class** (GPU/accelerator class
and memory class); it excludes hostnames, IP addresses, serials, ports, and local
paths.

## Limitations

- No published aggregate benchmark numbers.
- Unhealed: outputs may be degraded relative to the base model.
- Public weights are distributed through Hugging Face, not committed to GitHub.
- Private calibration sources and internal legal eval examples are not public.
- Surface fluency is not validation for legal use.
- All legal outputs require attorney review and source verification.

## Risk and Safety Considerations

Legal model compression can fail subtly: a pruned model may stay fluent while losing
source fidelity, refusal behavior, tool-selection judgment, or long-context
reasoning. Refusal/abstention was among the most-exposed capabilities at this ratio
in internal analysis, which is precisely why healing and held-out verification are
required gates. Any evaluation deployment should include source-grounding checks,
deterministic citation/document gates where possible, human legal review before any
client use or filing, prompt/source/output logging, and escalation when source
support is incomplete.

## Privacy Boundary

This public card intentionally excludes client names, closed-client materials,
attorney-client communications, privileged work product, private hostnames or file
paths, calibration row contents, prompts copied from live matters, and any secrets,
credentials, or tokens.

## License Boundary

This artifact is a derivative of MiniMax-M3 and remains subject to the **MiniMax
Community License**. The license permits use, modification, and redistribution
including publishing derivatives; commercial use additionally requires prominently
displaying **"Built with MiniMax M3"** and sending a notice to MiniMax (or, above a
revenue threshold, obtaining prior written authorization). A prohibited-uses
appendix applies. Review the upstream license in full before use, especially for
commercial deployment. The upstream `LICENSE` is redistributed with the Hugging Face
checkpoint.

Built with MiniMax M3.

## Version and Provenance

| Item | Value |
| --- | --- |
| Card version | 1.0 |
| Card date | 2026-07-24 |
| Public status | Research artifact — unhealed, under review |
| Related README | `README.md` |
| Method docs | `../../research/minimax-m3-legal-reap/` |
| Base model | `MiniMaxAI/MiniMax-M3` |
| Prune ratio / seed | 0.22 / 42 |
| Cut-plan hash | `<FILL: cut_plan_hash>` |
| Hugging Face checkpoint | `https://huggingface.co/ProprietaryLegal/minimax-m3-legal-reap-0.22` |

## Citation

> PLI Labs. MiniMax-M3 Legal REAP 0.22 model card. Proprietary Legal Intelligence,
> 2026. Derived from MiniMaxAI/MiniMax-M3. Pruning method after Lasby, Lazarevich,
> Sinnadurai, Lie, Ioannou, and Thangarasa, "REAP the Experts: Why Pruning Prevails
> for One-Shot MoE Compression", arXiv:2510.13999.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
- https://github.com/ProprietaryLegal/pli-labs
- https://huggingface.co/ProprietaryLegal/minimax-m3-legal-reap-0.22
- https://arxiv.org/abs/2510.13999
- https://huggingface.co/MiniMaxAI/MiniMax-M3
