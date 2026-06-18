# MiniMax-M2.7 Legal REAP 0.35

Status: benchmark-backed candidate.

Formal candidate model card: [MODEL_CARD.md](MODEL_CARD.md).

Hugging Face checkpoint:
[ProprietaryLegal/minimax-m2.7-legal-reap-0.35](https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.35).
GGUF release:
[ProprietaryLegal/minimax-m2.7-legal-reap-0.35-gguf](https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.35-gguf).

This page explains the MiniMax-M2.7 0.35 legal REAP as a legal-domain pruning
candidate. Current Opus-parity bench testing indicates the 0.35 prune is
performing well. See the model card for intended use, out-of-scope use,
validation status, limitations, and privacy boundaries.

## Why It Was Built

MiniMax-M2.7 was explored because it is a large mixture-of-experts model with
enough capacity to carry legal writing, source-grounded analysis, and long-form
reasoning. The goal was to test whether a legal calibration workload could
identify experts that mattered for legal work and remove experts that mattered
less, producing a smaller checkpoint suitable for local serving.

This is the opposite of blindly adopting a coding-focused prune. The point was
to preserve legal drafting, factual synthesis, refusal behavior, and
workflow-oriented reasoning rather than only preserving code performance.

## How The 0.35 Run Was Created

Known public-safe run facts:

- Base family: MiniMax-M2.7.
- Working precision: bf16 after FP8-to-bf16 pre-dequantization for V100-class
  hardware compatibility.
- Model structure used by the profile: 62 layers, 256 routed experts per layer,
  top-8 expert routing.
- Prune ratio: 0.35.
- Expert cut: 89 experts pruned per prunable layer.
- Remaining experts: 167 per prunable layer.
- Calibration shape: a legal-orchestrator composite with legal drafting and
  harness-style rows.
- Observed token count after the token-count fix: 1,310,184 non-padding tokens
  over 640 batches.
- The saved 0.35 artifact was verified structurally: 61 safetensors shards,
  index present, no missing indexed shard keys, and config showing 167 local
  experts.

## What It Proved

The 0.35 run proved useful engineering facts:

- FP8 source weights could be converted for V100-class hardware.
- A legal composite calibration set could drive a full observe/prune flow.
- REAP observations can be reused for multiple ratios.
- Structural pruning under CPU offload needed a safer live-state save path.
- Public release materials need validation evidence, not just successful model
  saving.

## What Current Testing Adds

Opus-parity bench testing now indicates that the 0.35 prune is performing well
on the legal comparison tasks being used for PLI model selection. That makes the
artifact stronger than a mere engineering demo, while still leaving one public
documentation task: publish a sanitized score summary that does not expose
private task names, matter details, or source examples.

## Release Locations

- Hugging Face checkpoint: https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.35
- Hugging Face GGUF: https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.35-gguf
- GitHub model card: MODEL_CARD.md

## Recommended Follow-Up

Add a public benchmark table with aggregate Opus-parity results, task-family
coverage, and limitations. If the 0.35 prune remains strong after the sanitized
score review, promote this page from candidate notes to a release-card format.

## Links

- PLI Labs website: https://proprietarylegal.com
- Research site: https://proprietarylegal.ai
