# Qwen3.5-104b-a10b-LegalReap

Formal candidate model card: [MODEL_CARD.md](MODEL_CARD.md).

Target Hugging Face checkpoint:
[ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap](https://huggingface.co/ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap).

## Summary

Qwen3.5-104b-a10b-LegalReap is PLI Labs' conservative legal-domain expert-pruned
Qwen3.5 candidate. It starts from Qwen3.5-122B-A10B and removes the stable low-saliency expert
core observed under legal calibration, reducing each MoE layer from 256 routed experts to 216
while preserving the original 48-layer depth, hidden size, tokenizer, and top-8 routing pattern.

The purpose is practical: make a frontier-scale open MoE more approachable for private legal AI
experimentation without pretending that smaller automatically means safer. The public release
is positioned as a research candidate for lawyer-supervised drafting, review, source-grounded
analysis, and local evaluation.

## Why This Cut

The Qwen3.5 run showed a flat saliency profile: this model is genuinely prune-resistant. PLI
Labs therefore did not treat aggressive pruning as the goal. The 0.16 release records the
stable droppable core rather than chasing a larger headline compression number.

Public-safe facts:

- Base model: Qwen3.5-122B-A10B.
- Estimated parameter count: about 104B.
- Method: REAP-style routed-expert pruning with survivor router renormalization.
- Expert count: 256 -> 216 routed experts per MoE layer.
- Layer count: 48 layers retained.
- Routing: top-8 routed experts retained.
- Release status: research candidate, not a validated production legal model.

## Links

- Hugging Face target: https://huggingface.co/ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap
- PLI Labs GitHub: https://github.com/ProprietaryLegal/pli-labs
- Model card: [MODEL_CARD.md](MODEL_CARD.md)
- Upload verification: Hugging Face file presence and byte sizes verified on
  2026-06-25; a separate checksum audit is not yet published.
