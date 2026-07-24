# REAP for Legal MoE Compression — Method Overview

This document explains, at a public-safe level, what REAP is, how it was applied to
MiniMax-M3, and why legal calibration differs from the coding calibration that most
open MoE-pruning work uses.

## What REAP is

REAP (Router-weighted Expert Activation Pruning) is a one-shot method for compressing
a mixture-of-experts model by **removing whole routed experts** rather than shrinking
each expert. For each expert, REAP measures a **saliency** signal that combines how
strongly the router gates that expert with how large its output activations are over
the tokens that actually route to it. Experts with the lowest saliency for the target
workload are pruned; the router's weights for the surviving experts are sliced down to
match. Reference: Lasby, Lazarevich, Sinnadurai, Lie, Ioannou, and Thangarasa, "REAP
the Experts: Why Pruning Prevails for One-Shot MoE Compression", arXiv:2510.13999.

Two consequences matter for a legal release:

1. **The calibration workload decides which experts look salient.** Prune with coding
   data and you preserve the experts coding uses. That is the wrong objective for a
   legal deployment.
2. **REAP by itself has no held-out quality gate** — it ranks and cuts to a chosen
   ratio. Whether a given ratio is *safe for legal work* is a separate question that a
   drop-ratio target does not answer.

## Why legal calibration is different

Coding benchmarks (e.g. pass@1 on programming tasks) reward a narrow, easily-measured
capability. Published deep MoE prunes that "hold at 40–50%" are almost always
validated on exactly those benchmarks. But legal work lives or dies on **knowledge,
source fidelity, faithfulness, refusal/abstention, and long-context reasoning** — and
these degrade *before* surface fluency does. A pruned model can still write smooth
legal-sounding prose while inventing a citation or a date. That failure mode is a
knowledge/faithfulness failure, invisible to a coding benchmark.

So the safe prune depth for a legal MoE is governed by the **worst** capability axis,
not by average perplexity or a coding score. For MiniMax-M3 on legal work, internal
analysis places the honest cold-drop safe band at roughly **0.16–0.25**, with refusal
and long-context faithfulness the most exposed.

## Why this artifact is at 0.22, adaptive, and floor-enforced

A single uniform drop ratio forces bad cuts on the few sensitive early MoE layers
while leaving separable deeper layers under-pruned. Instead this artifact uses a
**per-layer adaptive schedule**: the overall budget is 0.22, but each layer's retained
count is set individually (89–111 of 128 experts) so that separable layers give up
more and capability-carrying layers are protected. **Capability floors** guarantee no
under-covered layer and no low-confidence carrier is cut below a safe count. The dense
layers 0–2 and the always-on shared expert are never touched.

MiniMax-M3's experts were also found to be **near-orthogonal in weight space** — there
is essentially no redundancy to exploit by merging experts instead of dropping them.
That is why the deployable path is adaptive **drop** plus a recovery **heal**, not a
merge.

## Why it is released unhealed and unevaluated

Recovery healing (a fine-tune that recovers capability lost to the cut) is the next
pipeline stage; held-out KL / faithfulness verification against the base model is the
gate after that. This artifact is the raw prune, published as a **research artifact**
so the method and the checkpoint can be inspected. Public quality claims wait for the
heal result and the verification tables. A recovery heal typically buys about one
ratio rung of capability back — it is not a substitute for choosing a safe base ratio,
and it can mask calibration-domain damage while held-out capability stays degraded,
which is exactly why verification is held-out.

## The measurement, honestly framed

The proxies that pick which ratios are worth GPU time (saliency spread, live-capacity
flags) are **triage inputs, not sign-off**. They tend to over-flag: they measure
uniform-drop safety and are blind to redundancy. Sign-off for a legal release is
intended to be **empirical** — held-out KL-from-base and top-1 flip-rate as primary
signals, with hard floors on date/money/citation recall and zero dispositive flips —
not a proxy heuristic. This artifact has passed the triage and capability-floor gates;
it has **not** passed the empirical sign-off, which is why it is under review.
