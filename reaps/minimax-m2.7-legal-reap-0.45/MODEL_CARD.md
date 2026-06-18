# Model Card: MiniMax-M2.7 Legal REAP 0.45

## Summary

MiniMax-M2.7 Legal REAP 0.45 is an aggressive benchmark-backed candidate
produced by PLI Labs to test how far legal-domain expert pruning can compress a
large mixture-of-experts model before legal capability breaks. It uses the same
legal-orchestrator direction as the 0.35 candidate, but removes more experts
per prunable layer.

This is a candidate model card. The public repository records the method,
known structural facts, intended use, limitations, and validation status. Public
aggregate benchmark tables should be added before this card is treated as a
validated release card.

## Model Details

| Field | Value |
| --- | --- |
| Model family | MiniMax-M2.7 |
| Release name | MiniMax-M2.7 Legal REAP 0.45 |
| Release status | Benchmark-backed candidate |
| Publisher | PLI Labs / Proprietary Legal Intelligence |
| Pruning method | REAP-style routed-expert pruning |
| Base architecture profile | 62 layers, 256 routed experts per layer, top-8 routing |
| Working precision | bf16 after FP8-to-bf16 pre-dequantization for V100-class compatibility |
| Prune ratio | 0.45 |
| Expert cut | Approximately 115 experts pruned per prunable layer |
| Remaining experts | Approximately 141 experts per prunable layer |
| Public weights | Distributed through Hugging Face |
| Public docs | `README.md`, this model card, and the MiniMax-M2.7 campaign narrative |
| Contact | nick@proprietarylegal.com |

## Intended Use

This candidate is intended for research and evaluation of aggressive
legal-domain model compression. The target workflows are:

- legal drafting and revision assistance under lawyer supervision;
- long-form legal and factual synthesis;
- source-grounded summarization;
- retrieval-sensitive analysis;
- conservative refusal behavior when the record is incomplete;
- workflow-oriented agent behavior in legal automation harnesses;
- local serving experiments where a smaller checkpoint may materially improve
  fit, context, or throughput.

The intended deployment pattern is local or on-premises evaluation on hardware
controlled by the lawyer, firm, or legal organization.

## Out-of-Scope Use

This candidate should not be used for:

- unsupervised legal advice;
- final court filings without attorney review;
- citation generation without independent source verification;
- client-specific work where the user cannot verify source support;
- production deployment based only on this card;
- claims that aggressive pruning is safe for all legal tasks;
- non-legal capability claims such as coding superiority or general chat
  quality.

## Calibration Signal

The calibration shape followed the same legal-orchestrator direction used in
the MiniMax-M2.7 campaign. It was designed around legal drafting plus
harness-style behavior, with the goal of preserving legal prose, factual
synthesis, source fidelity, refusal behavior, and workflow-sensitive reasoning.

The public repository does not include private prompts, client matter details,
privileged work product, private file paths, or raw client-source examples.

## Pruning Procedure

The run used REAP-style routed-expert pruning:

1. Convert or prepare MiniMax-M2.7 weights for the target working precision.
2. Run the legal calibration workload through the observer.
3. Rank routed experts by observed saliency for the legal workload.
4. Remove low-saliency experts according to the 0.45 prune ratio.
5. Save a smaller checkpoint with approximately 141 local experts per prunable
   layer.
6. Compare the aggressive lane against the 0.35 candidate and base-model
   behavior on legal tasks.

The 0.45 candidate is intentionally a pressure test. It asks whether legal
quality survives a deeper cut than the 0.35 lane.

## Evaluation Status

Current status: benchmark-backed candidate.

PLI Labs internal Opus-parity bench testing indicates that the 0.45 prune is
performing well despite the more aggressive expert cut. That makes it a
commercially important candidate because smaller local serving cost only matters
if legal capability survives. Public aggregate score tables are still pending
and should be added before the candidate is promoted to a validated release.

The next public validation update should include:

- aggregate Opus-parity score summaries;
- task-family coverage;
- comparison against the 0.35 prune and base model where available;
- known failures or regressions;
- recommended deployment envelope;
- any areas where 0.45 trades breadth for smaller local serving cost.

## Hardware and Serving Notes

The candidate was designed for local owned-hardware evaluation. Deployment
should be validated on the intended runtime, quantization path, context tier,
and legal workflow before any production use.

Because 0.45 is the more aggressive candidate, serving gains should be evaluated
beside legal quality results. Smaller is useful only if the legal capability
survives.

PLI Labs' public owned-hardware research is relevant background. The public
research pages summarize broad serving lessons without exposing private launch
recipes, internal topology, hostnames, ports, or local file paths:

- `../../research/v100/README.md`
- `../../research/b70/README.md`

## Limitations

- Public aggregate benchmark numbers are not yet attached to this card.
- Public weights are distributed through Hugging Face, not committed to GitHub.
- Private calibration sources and internal legal eval examples are not public.
- The card records a candidate, not a production-ready legal advice system.
- Aggressive pruning can preserve fluency while damaging source fidelity or
  legal judgment.
- All legal outputs require attorney review and source verification.

## Risk and Safety Considerations

The 0.45 candidate has a higher compression target than 0.35, so validation risk
is higher. A pruned model may remain fluent while losing refusal behavior,
tool-selection judgment, retrieval-sensitive reasoning, or long-context
stability. This card therefore treats the candidate as benchmark-backed but not
fully validated.

Any deployment should include:

- source-grounding checks;
- deterministic document and citation gates where possible;
- human legal review before client use or filing;
- logging of prompts, sources, and outputs for auditability;
- refusal or escalation behavior when source support is incomplete;
- regression testing against the 0.35 lane and the base model.

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
| Hugging Face checkpoint | `https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45` |
| Hugging Face GGUF | `https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45-gguf` |

## Citation

If discussing this candidate, cite it as:

> PLI Labs. MiniMax-M2.7 Legal REAP 0.45 model card. Proprietary Legal
> Intelligence, 2026.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
- https://github.com/ProprietaryLegal/pli-labs
- https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45
- https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45-gguf
