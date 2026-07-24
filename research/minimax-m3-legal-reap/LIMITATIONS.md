# Limitations — MiniMax-M3 Legal REAP 0.22

This artifact is an **unhealed research prune**. Its limitations are substantial and
stated plainly.

## Status limitations

- **No published evaluation.** There are no aggregate benchmark tables. Nothing here
  should be read as a quality claim.
- **Unhealed.** This is the raw prune, before any recovery fine-tune. Capability lost
  to the cut has not been recovered. Outputs may be degraded relative to the base
  model.
- **Verification pending.** The intended sign-off gate — held-out KL-from-base and
  top-1 flip-rate against the base model, with hard floors on date/money/citation
  recall and zero dispositive flips — has **not** been run on this checkpoint.
- **Held under review.** The artifact was produced as a first-of-kind research output
  and is flagged for review, not release.

## Capability limitations (expected failure modes)

Deep MoE expert pruning tends to preserve surface fluency while degrading the
capabilities that legal work depends on. For this artifact, expect the largest risk on:

- **Knowledge and source fidelity** — plausible-but-wrong citations, dates, dollar
  amounts, or holdings. This is the failure mode that matters most for legal use and
  the one coding-style benchmarks do not detect.
- **Refusal / abstention** — internal analysis identified refusal behavior as among
  the most-exposed capabilities at this ratio. A pruned model may answer where it
  should decline or escalate.
- **Long-context reasoning** — faithfulness over long records can degrade even when
  short-context behavior looks intact.
- **Tool / workflow selection** — judgment about which action to take in an
  agentic legal harness.

## Scope limitations

- **Text-only.** The vision tower and multimodal projector are removed; no multimodal
  capability is present or claimed.
- **Serving is immature.** MiniMax Sparse Attention has no current Ampere/Volta kernel,
  so serving today generally means a text-only dense-fallback path as backends mature.
  The artifact's value is durable, portable, pruned weights.
- **Not a benchmark of the base model.** No comparison numbers against
  `MiniMaxAI/MiniMax-M3` are published yet.

## Use limitations

- Research and lawyer-supervised evaluation only.
- Not for production, unsupervised legal advice, final filings without attorney review,
  or citation generation without independent verification.
- All legal outputs require attorney review and independent source verification.

## What would lift these limitations

A recovery heal, followed by held-out KL / faithfulness verification and published
aggregate legal task-family scores (multiple seeds, verified served identity, base-model
comparison, and a stated deployment envelope). Until then, this is a research artifact.
