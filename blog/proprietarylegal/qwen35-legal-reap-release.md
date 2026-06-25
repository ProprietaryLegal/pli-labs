# PLI Labs Releases Qwen3.5 Legal REAP Models For Lawyer-Supervised Evaluation

PLI Labs has published two public Qwen3.5 legal-compression candidates:

- [Qwen3.5-122B-A10B Legal REAP 0.16](https://huggingface.co/ProprietaryLegal/qwen35-122b-a10b-legal-reap-0.16)
- [Qwen3.5-122B-A10B Legal REAP 0.16 LayerDrop-6](https://huggingface.co/ProprietaryLegal/qwen35-122b-a10b-legal-reap-0.16-layerdrop-6)

These releases are part of Proprietary Legal Intelligence's practical research
program: build legal AI systems that lawyers can inspect, evaluate, and run in
private settings, then publish enough of the method for serious technical review.

## What Was Released

The first artifact, Legal REAP 0.16, is a conservative expert-reduced checkpoint
derived from Qwen3.5-122B-A10B. It reduces each MoE layer from 256 routed experts
to 216 while preserving 48 layers, the tokenizer, hidden size, and top-8 routing
pattern.

The second artifact, Legal REAP 0.16 LayerDrop-6, is a smaller companion model.
It starts from the 0.16 REAP checkpoint and removes six decoder layers: 8, 9,
12, 13, 16, and 17. That gives evaluators a second compression axis to test:
expert reduction plus depth reduction.

## Why A Legal REAP

Most open model compression work is judged against generic or coding-heavy
benchmarks. Legal work stresses different behavior. A useful legal assistant has
to preserve factual care, source fidelity, drafting judgment, refusal discipline,
and workflow awareness.

PLI Labs built this release around that premise. The goal was not to create an
autonomous lawyer. The goal was to produce research artifacts for
lawyer-supervised evaluation of a frontier-scale open MoE model compressed
around legal work.

## Why The Cut Is Conservative

The Qwen3.5 run showed a flatter expert-saliency profile than expected. In plain
terms: this model resisted obvious aggressive pruning. That is an important
technical result. Instead of forcing a larger compression claim, PLI Labs
published the stable 0.16 expert cut and a separate LayerDrop derivative.

That makes the release more useful. Legal AI does not need inflated compression
headlines. It needs inspectable models, clear caveats, and evaluation paths that
legal teams can reproduce.

## Public Documentation

The supporting documentation is public in the PLI Labs GitHub repository:

- [Qwen3.5 legal REAP research packet](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/qwen35-legal-reap)
- [Stack and settings summary](https://github.com/ProprietaryLegal/pli-labs/blob/main/research/qwen35-legal-reap/STACK_AND_SETTINGS.md)
- [Legal REAP 0.16 model card](https://github.com/ProprietaryLegal/pli-labs/blob/main/reaps/qwen35-122b-a10b-legal-reap-0.16/MODEL_CARD.md)
- [LayerDrop-6 model card](https://github.com/ProprietaryLegal/pli-labs/blob/main/reaps/qwen35-122b-a10b-legal-reap-0.16-layerdrop-6/MODEL_CARD.md)

The public materials intentionally do not identify private legal source items,
client names, matter names, private prompts, attorney-client communications, or
privileged work product.

## Intended Use

These models are research candidates for lawyer-supervised evaluation. They are
appropriate for controlled testing of legal drafting, revision, summarization,
workflow behavior, and source-grounded analysis. They are not unsupervised legal
advice systems, and they are not a substitute for attorney review.

PLI Labs is optimistic about this direction because legal AI should become more
private, more inspectable, and more aligned with the actual work lawyers do. The
release is conservative because legal reliability has to be earned with evidence.
