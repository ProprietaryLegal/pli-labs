# Research Notes: What The Qwen3.5 Legal REAP Run Shows

PLI Labs' Qwen3.5 legal REAP release is built around a simple research question:
can a very large open mixture-of-experts model be reduced around legal-work
behavior without treating coding benchmarks as the center of the world?

The public release includes:

- [Qwen3.5-122B-A10B Legal REAP 0.16](https://huggingface.co/ProprietaryLegal/qwen35-122b-a10b-legal-reap-0.16)
- [Qwen3.5-122B-A10B Legal REAP 0.16 LayerDrop-6](https://huggingface.co/ProprietaryLegal/qwen35-122b-a10b-legal-reap-0.16-layerdrop-6)

## The Method

REAP-style expert pruning observes which routed experts activate under a
calibration workload, ranks low-saliency experts, removes the stable droppable
core, and renormalizes the router over surviving experts.

For this run, the calibration direction was legal work: long-form synthesis,
source-grounded review, legal workflow prompts, structured outputs, refusal
behavior, and authored legal reasoning. PLI Labs does not publish the private
calibration items or identify the underlying legal source materials.

## The Main Finding

The most important result is not just that a checkpoint was produced. The
important result is that Qwen3.5-122B-A10B appears prune-resistant.

The saliency profile did not support a confident aggressive expert cut. PLI Labs
therefore published a conservative 0.16 release: each MoE layer goes from 256
routed experts to 216, while retaining the original 48-layer depth and top-8
routing pattern.

This matters because legal AI compression should not be driven by the desire for
a clean marketing number. If the model resists pruning, the correct research
answer is to say so and preserve the legal behavior that appears most important.

## The Companion LayerDrop Model

The LayerDrop-6 checkpoint asks a different question. After the conservative
expert prune, can depth be reduced in a controlled way?

The companion model removes layers 8, 9, 12, 13, 16, and 17 from the REAP-0.16
checkpoint, reducing the model from 48 layers to 42 while retaining 216 routed
experts per remaining MoE layer.

That does not prove the smaller model is better. It gives legal evaluators a
clean comparison point: base model versus expert-reduced model versus
expert-reduced plus depth-reduced model.

## What To Evaluate Next

Before anyone treats these checkpoints as production legal models, the right
next tests are practical legal tests:

- source-grounded summarization with answerable and unanswerable prompts;
- citation discipline, where every material claim must be supported by a source;
- refusal behavior when the record is incomplete;
- drafting and revision quality under lawyer supervision;
- long-context degradation;
- hallucinated-authority resistance;
- quantized serving behavior on the target hardware.

## Why Publish It

PLI Labs is publishing these models and notes because legal AI should not depend
only on closed systems or generic model cards. Lawyers and legal technologists
need inspectable artifacts, clear limitations, and model releases that explain
what legal behavior the work tried to preserve.

This release is optimistic about the direction and conservative about the claim:
Qwen3.5 legal REAP is a serious research candidate for legal-work-oriented model
compression, not an autonomous legal advice product.
