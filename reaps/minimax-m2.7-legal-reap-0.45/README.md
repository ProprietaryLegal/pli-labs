# MiniMax-M2.7 Legal REAP 0.45

Status: benchmark-backed candidate.

Formal candidate model card: [MODEL_CARD.md](MODEL_CARD.md).

Hugging Face checkpoint:
[ProprietaryLegal/minimax-m2.7-legal-reap-0.45](https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45).
GGUF release:
[ProprietaryLegal/minimax-m2.7-legal-reap-0.45-gguf](https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45-gguf).

This page explains the MiniMax-M2.7 0.45 legal REAP as an aggressive
legal-domain pruning candidate. Current Opus-parity bench testing indicates the
0.45 prune is performing well. See the model card for intended use,
out-of-scope use, validation status, limitations, and privacy boundaries.

## Why It Was Built

The 0.45 run asked a harder question than the 0.35 run: how far can a legal
calibration workload compress a large MoE model before legal capability breaks?

That question matters because local lawyers and firms care about practical
serving cost. A smaller checkpoint can fit on more modest hardware, run at
higher context, and become easier to finetune. But an aggressive prune is only
valuable if the legal skills survive.

## How The 0.45 Run Was Created

Known public-safe run facts:

- Base family: MiniMax-M2.7.
- Working precision: bf16 after FP8-to-bf16 pre-dequantization for V100-class
  hardware compatibility.
- Model structure used by the profile: 62 layers, 256 routed experts per layer,
  top-8 expert routing.
- Prune ratio: 0.45.
- Expert cut: approximately 115 experts pruned per prunable layer.
- Remaining experts: approximately 141 per prunable layer.
- Calibration shape: the same legal-orchestrator direction used in the MiniMax
  campaign, designed around legal drafting plus harness-style behavior.
- Evaluation intent: compare the deeper prune against the 0.35 lane and the base
  model on legal tasks, including long-context and workflow-sensitive probes.

## What It Proved

The 0.45 lane is useful as a pressure test. It shows why a legal REAP cannot be
judged by size reduction alone. A model can remain fluent while losing the
capabilities lawyers actually need: careful legal prose, source fidelity,
tool-selection judgment, retrieval-sensitive behavior, and long-context
reasoning.

## What Current Testing Adds

Opus-parity bench testing now indicates that the 0.45 prune is performing well
despite the more aggressive expert cut. That is the commercially interesting
result: if legal quality holds at 0.45, the model may be materially easier to
serve locally while preserving the legal capabilities PLI cares about.

## Release Locations

- Hugging Face checkpoint: https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45
- Hugging Face GGUF: https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45-gguf
- GitHub model card: MODEL_CARD.md

## Recommended Follow-Up

Before a formal release, add a sanitized benchmark table that covers:

- aggregate Opus-parity performance;
- task-family coverage;
- comparison against the 0.35 prune and the base model where available;
- limitations and recommended use cases;
- any areas where 0.45 trades breadth for smaller local serving cost.

## Links

- PLI Labs website: https://proprietarylegal.com
- Research site: https://proprietarylegal.ai
