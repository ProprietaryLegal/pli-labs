# Model Card: MiniMax-M2.7 Legal REAP 0.35

## Summary

MiniMax-M2.7 Legal REAP 0.35 is a benchmark-backed candidate produced by PLI
Labs to test legal-domain expert pruning for a large mixture-of-experts model.
The candidate uses a legal calibration direction rather than a coding benchmark:
the pruning target is preservation of legal drafting, long-form factual
synthesis, source-grounded analysis, refusal behavior, and workflow-oriented
reasoning.

This is a candidate model card. The public repository records the method,
known structural facts, intended use, limitations, and validation status. Public
aggregate benchmark tables should be added before this card is treated as a
validated release card.

## Model Details

| Field | Value |
| --- | --- |
| Model family | MiniMax-M2.7 |
| Release name | MiniMax-M2.7 Legal REAP 0.35 |
| Release status | Benchmark-backed candidate |
| Publisher | PLI Labs / Proprietary Legal Intelligence |
| Pruning method | REAP-style routed-expert pruning |
| Base architecture profile | 62 layers, 256 routed experts per layer, top-8 routing |
| Working precision | bf16 after FP8-to-bf16 pre-dequantization for V100-class compatibility |
| Prune ratio | 0.35 |
| Expert cut | 89 experts pruned per prunable layer |
| Remaining experts | 167 experts per prunable layer |
| Public weights | Not distributed from this repository as of this card |
| Public docs | `README.md`, this model card, and the MiniMax-M2.7 campaign narrative |
| Contact | nick@proprietarylegal.com |

## Intended Use

This candidate is intended for research and evaluation of legal-domain model
compression. The target workflows are:

- legal drafting and revision assistance under lawyer supervision;
- long-form legal and factual synthesis;
- source-grounded summarization;
- retrieval-sensitive analysis;
- conservative refusal behavior when the record is incomplete;
- workflow-oriented agent behavior in legal automation harnesses.

The intended deployment pattern is local or on-premises evaluation on hardware
controlled by the lawyer, firm, or legal organization.

## Out-of-Scope Use

This candidate should not be used for:

- unsupervised legal advice;
- final court filings without attorney review;
- citation generation without independent source verification;
- client-specific work where the user cannot verify source support;
- production deployment based only on this card;
- non-legal capability claims such as coding superiority or general chat
  quality.

## Calibration Signal

The calibration shape was a legal-orchestrator composite with legal drafting
rows and harness-style rows. The intent was to observe which routed experts were
used by legal drafting, source-grounded synthesis, refusal behavior, and
workflow-sensitive legal tasks.

Known public-safe calibration facts:

- observed token count after the token-count fix: 1,310,184 non-padding tokens;
- calibration batch count: 640 batches;
- calibration direction: legal drafting plus harness-style legal behavior;
- optimization target: preserve legal capability rather than coding benchmark
  performance.

The public repository does not include private prompts, client matter details,
privileged work product, private file paths, or raw client-source examples.

## Pruning Procedure

The run used REAP-style routed-expert pruning:

1. Convert or prepare MiniMax-M2.7 weights for the target working precision.
2. Run the legal calibration workload through the observer.
3. Rank routed experts by observed saliency for the legal workload.
4. Remove low-saliency experts according to the 0.35 prune ratio.
5. Save a smaller checkpoint with 167 local experts per prunable layer.
6. Verify that the saved artifact has a coherent shard/index/config structure.

The 0.35 candidate was structurally verified with:

- 61 safetensors shards;
- index present;
- no missing indexed shard keys;
- config showing 167 local experts.

## Evaluation Status

Current status: benchmark-backed candidate.

PLI Labs internal Opus-parity bench testing indicates that the 0.35 prune is
performing well on the legal comparison tasks used for model selection. Public
aggregate score tables are still pending and should be added before the
candidate is promoted to a validated release.

The next public validation update should include:

- aggregate Opus-parity score summaries;
- task-family coverage;
- comparison against the 0.45 prune and base model where available;
- known failures or regressions;
- recommended deployment envelope;
- limitations that were discovered during legal evaluation.

## Hardware and Serving Notes

The run was designed with V100-class compatibility in mind by using bf16 after
FP8-to-bf16 pre-dequantization. This does not mean every V100 serving backend
will run the candidate efficiently. V100-class deployment should be tested on
the exact backend, quantization path, context length, and GPU topology intended
for production use.

PLI Labs' public V100 research is relevant background:

- `../../research/v100/README.md`
- `../../research/v100/findings/current-stack.md`
- `../../research/v100/findings/software-stack.md`
- `../../research/v100/testing/benchmark-summary.md`

## Limitations

- Public aggregate benchmark numbers are not yet attached to this card.
- Public weights are not distributed from this repository as of this card.
- Private calibration sources and internal legal eval examples are not public.
- The card records a candidate, not a production-ready legal advice system.
- Strong surface fluency is not sufficient validation for legal use.
- All legal outputs require attorney review and source verification.

## Risk and Safety Considerations

Legal model compression can fail in subtle ways. A pruned model may remain
fluent while losing source fidelity, refusal behavior, tool-selection judgment,
or long-context reasoning. This card therefore treats the candidate as
benchmark-backed but not fully validated.

Any deployment should include:

- source-grounding checks;
- deterministic document and citation gates where possible;
- human legal review before client use or filing;
- logging of prompts, sources, and outputs for auditability;
- refusal or escalation behavior when source support is incomplete.

## Privacy Boundary

This public card intentionally excludes:

- client names;
- closed-client materials;
- attorney-client communications;
- privileged work product;
- private hostnames or file paths;
- raw prompts copied from live matters;
- secrets, credentials, or tokens.

## Version and Provenance

| Item | Value |
| --- | --- |
| Card version | 1.0 |
| Card date | 2026-06-18 |
| Public status | Benchmark-backed candidate |
| Related README | `README.md` |
| Campaign narrative | `../../wiki/minimax-m2.7-legal-reap-campaign.md` |
| Release principles | `../../wiki/release-principles.md` |

## Citation

If discussing this candidate, cite it as:

> PLI Labs. MiniMax-M2.7 Legal REAP 0.35 model card. Proprietary Legal
> Intelligence, 2026.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
- https://github.com/ProprietaryLegal/pli-labs
