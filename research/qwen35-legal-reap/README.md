# Qwen3.5 Legal REAP Research Packet

This packet documents PLI Labs' Qwen3.5-122B-A10B legal REAP release line.

The release has two public artifacts:

- [Qwen3.5-122B-A10B Legal REAP 0.16](https://huggingface.co/ProprietaryLegal/qwen35-122b-a10b-legal-reap-0.16)
- [Qwen3.5-122B-A10B Legal REAP 0.16 LayerDrop-6](https://huggingface.co/ProprietaryLegal/qwen35-122b-a10b-legal-reap-0.16-layerdrop-6)

The first model is the conservative expert-reduced checkpoint. The second model
is a depth-reduced companion derived from the first checkpoint. PLI Labs is not
claiming that either artifact is a validated production legal model; both are
research candidates for lawyer-supervised evaluation.

## Research Thesis

Legal work stresses capabilities that generic compression runs often fail to
measure: long-form synthesis, source fidelity, refusal discipline, factual
care, tool-aware workflow behavior, and the ability to draft in a lawyer's
actual working style while remaining verifiable. PLI Labs' position is that
large open MoE models should be compressed around those legal behaviors when
the intended deployment is legal work.

The Qwen3.5 run produced a useful negative result: the model was not an obvious
candidate for aggressive expert removal. The saliency profile was flatter than
expected, so PLI Labs chose a conservative 0.16 cut that records the stable
droppable expert core instead of forcing a larger compression headline.

## Released Artifacts

| Artifact | Summary | HF repo |
| --- | --- | --- |
| Legal REAP 0.16 | 48-layer Qwen3.5 MoE checkpoint with 216 routed experts per layer | [qwen35-122b-a10b-legal-reap-0.16](https://huggingface.co/ProprietaryLegal/qwen35-122b-a10b-legal-reap-0.16) |
| Legal REAP 0.16 LayerDrop-6 | 42-layer derivative with layers 8, 9, 12, 13, 16, and 17 removed | [qwen35-122b-a10b-legal-reap-0.16-layerdrop-6](https://huggingface.co/ProprietaryLegal/qwen35-122b-a10b-legal-reap-0.16-layerdrop-6) |

## Public Documentation

- [STACK_AND_SETTINGS.md](STACK_AND_SETTINGS.md) - public-safe run settings,
  architecture facts, and artifact inventory.
- [UPLOAD_STATUS.md](UPLOAD_STATUS.md) - publication targets and upload notes.
- [reaps/qwen35-122b-a10b-legal-reap-0.16](../../reaps/qwen35-122b-a10b-legal-reap-0.16/README.md)
- [reaps/qwen35-122b-a10b-legal-reap-0.16-layerdrop-6](../../reaps/qwen35-122b-a10b-legal-reap-0.16-layerdrop-6/README.md)
- [wiki/qwen35-legal-reap-campaign.md](../../wiki/qwen35-legal-reap-campaign.md)

## Privacy Boundary

This packet intentionally does not identify private legal source items,
calibration item titles, client names, matter names, attorney-client
communications, privileged work product, local hostnames, private filesystem
paths, credentials, or tokens.

The public release documents the method and artifact facts. The private working
repository remains the source of raw logs and internal run details.

## Recommended Evaluation Before Production Use

Before treating either checkpoint as production-ready, legal evaluators should
run:

- source-grounded summarization tests with answerable and unanswerable prompts;
- citation discipline tests where source text must support each proposition;
- refusal/escalation checks for missing record support;
- style-transfer tests against approved lawyer work product;
- long-context degradation checks;
- hallucinated-authority checks;
- latency, memory, and quantization tests on the target serving stack.

PLI Labs' release posture is optimistic but conservative: these artifacts are
promising legal-compression candidates, not substitutes for legal judgment.
