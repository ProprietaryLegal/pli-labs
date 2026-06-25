# Qwen3.5 Legal REAP Campaign

PLI Labs' Qwen3.5 legal REAP campaign is a public research release for
lawyer-supervised legal AI: take a very large open MoE model, compress it around
legal-work behavior rather than coding behavior, and publish enough detail for
other legal AI builders to inspect the method.

## Public Release

Two public model repositories are part of this campaign:

- [Qwen3.5-104b-a10b-LegalReap](https://huggingface.co/ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap)
- [Qwen3.5-91b-a10b-LegalReap-Layerdrop6](https://huggingface.co/ProprietaryLegal/Qwen3.5-91b-a10b-LegalReap-Layerdrop6)

The first model is an expert-reduced REAP checkpoint. It reduces each MoE layer
from 256 routed experts to 216 while retaining 48 layers and top-8 routing. The
second model is a depth-reduced companion that removes six decoder layers from
the REAP checkpoint.

## Why It Matters

Legal AI needs models that can be evaluated against legal work, not only generic
benchmarks. The target behaviors include:

- source-grounded analysis;
- careful factual synthesis;
- long-form legal drafting and revision;
- refusal discipline when the record is incomplete;
- workflow-sensitive tool use;
- private or owned-hardware evaluation.

Qwen3.5-122B-A10B was not an obvious aggressive-pruning target. The run showed a
flatter saliency profile than expected, which is a meaningful finding by itself.
PLI Labs therefore chose a conservative 0.16 expert cut instead of overstating a
larger compression result.

## Public Technical Summary

| Item | Qwen3.5-104b-a10b-LegalReap | Qwen3.5-91b-a10b-LegalReap-Layerdrop6 |
| --- | --- | --- |
| Base | Qwen/Qwen3.5-122B-A10B | Legal REAP 0.16 |
| Estimated parameters | about 104B | about 91B |
| Layers | 48 | 42 |
| Routed experts per layer | 216 | 216 |
| Original experts per layer | 256 | 256 |
| Experts per token | 8 | 8 |
| Release posture | Research candidate | Research candidate |

The Layerdrop6 checkpoint removes layers 8, 9, 12, 13, 16, and 17.

## Boundaries

The public campaign does not identify private legal source items, client names,
matter names, private prompts, attorney-client communications, privileged work
product, local hostnames, local paths, credentials, or tokens.

The release is promotional because the direction matters: lawyers should have
models that can be inspected, reduced, and evaluated for legal work. It is also
careful because legal fluency is not legal reliability. Both models require
attorney supervision and independent source verification.

## Supporting Pages

- [Research packet](../research/qwen35-legal-reap/README.md)
- [Stack and settings](../research/qwen35-legal-reap/STACK_AND_SETTINGS.md)
- [Qwen3.5-104b-a10b-LegalReap model card](../reaps/Qwen3.5-104b-a10b-LegalReap/MODEL_CARD.md)
- [Qwen3.5-91b-a10b-LegalReap-Layerdrop6 model card](../reaps/Qwen3.5-91b-a10b-LegalReap-Layerdrop6/MODEL_CARD.md)
