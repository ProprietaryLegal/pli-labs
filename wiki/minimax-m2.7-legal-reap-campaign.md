# MiniMax-M2.7 Legal REAP Campaign

This page is the public campaign narrative for PLI Labs' MiniMax-M2.7 legal
REAP work.

The short version: PLI Labs tested whether a large open mixture-of-experts model
could be pruned around legal workloads instead of coding workloads. Current
Opus-parity bench testing indicates that both the 0.35 and 0.45 prunes are
performing well. The next public step is to publish sanitized aggregate
benchmark summaries.

## Why This Campaign Matters

Most public model pruning work is optimized around coding, general chat, or
generic benchmark performance. That is useful, but it is not the same thing as
legal work.

Legal drafting and review stress different capabilities:

- long-form factual synthesis;
- formal legal prose;
- careful distinction between facts, inferences, and advocacy;
- source-grounded summarization;
- retrieval-sensitive reasoning;
- conservative refusal behavior;
- lawyer-reviewable intermediate outputs.

A coding-focused prune can look efficient while removing the exact capabilities
a lawyer needs. PLI's legal REAP thesis is that the calibration workload should
match the legal behavior the model is supposed to preserve.

## What Was Built

The MiniMax-M2.7 campaign used a legal-orchestrator calibration direction rather
than a coding benchmark. Public-safe run facts:

- Base family: MiniMax-M2.7.
- Working precision: bf16 after FP8-to-bf16 pre-dequantization for V100-class
  hardware compatibility.
- Architecture profile: 62 layers, 256 routed experts per layer, top-8 routing.
- Calibration shape: legal drafting rows plus harness-style agentic and refusal
  behavior rows.
- 0.35 prune: 89 experts pruned per prunable layer, 167 experts remaining.
- 0.45 prune: approximately 115 experts pruned per prunable layer,
  approximately 141 experts remaining.

The 0.35 artifact was structurally verified with 61 safetensors shards, an index
file, no missing indexed shard keys, and configuration showing 167 local experts.

## Why 0.35 And 0.45 Are Both Interesting

The 0.35 prune asks whether a large MoE model can be made materially smaller
while preserving legal capability. It is the more conservative candidate.

The 0.45 prune asks whether a more aggressive expert cut can still hold up on
legal work. That is commercially important because a smaller checkpoint can be
easier to serve locally, easier to evaluate repeatedly, and easier to build on
with workflow-specific finetuning.

PLI Labs treats both as benchmark-backed candidates until the public benchmark
summary is sanitized and published.

## Public Positioning

The public story should be direct:

- This is legal-domain model compression, not generic compression.
- The purpose is to preserve legal writing, source fidelity, long-context
  synthesis, and lawyer-verifiable output.
- Coding skill is not the primary optimization target.
- Smaller is useful only if the legal capability survives.
- Current internal Opus-parity testing is positive, but public release pages
  should include sanitized aggregate scores before becoming formal model cards.

## README And Release Use

Use this page as the shared explanation for:

- [MiniMax-M2.7 legal REAP 0.35](../reaps/minimax-m2.7-legal-reap-0.35/README.md)
- [MiniMax-M2.7 legal REAP 0.45](../reaps/minimax-m2.7-legal-reap-0.45/README.md)
- future legal REAP model cards;
- future legal finetunes built on pruned checkpoints.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
