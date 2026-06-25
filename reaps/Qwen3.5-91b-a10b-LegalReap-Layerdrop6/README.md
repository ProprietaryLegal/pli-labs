# Qwen3.5-91b-a10b-LegalReap-Layerdrop6

Formal candidate model card: [MODEL_CARD.md](MODEL_CARD.md).

Target Hugging Face checkpoint:
[ProprietaryLegal/Qwen3.5-91b-a10b-LegalReap-Layerdrop6](https://huggingface.co/ProprietaryLegal/Qwen3.5-91b-a10b-LegalReap-Layerdrop6).

## Summary

Qwen3.5-91b-a10b-LegalReap-Layerdrop6 is the depth-reduced companion to PLI Labs'
conservative Qwen3.5 expert prune. It starts from the REAP-0.16 checkpoint and removes six
low-impact decoder layers selected by a calibration-driven layer-importance pass.

This release explores a second compression axis. REAP reduces routed expert capacity; layerdrop
reduces depth. The result keeps 216 routed experts per remaining MoE layer and reduces the
network from 48 to 42 layers.

## Public-Safe Technical Facts

- Parent checkpoint: Qwen3.5-104b-a10b-LegalReap.
- Estimated parameter count: about 91B.
- Method: calibration-driven whole-layer drop.
- Dropped layer indices: 8, 9, 12, 13, 16, 17.
- Remaining layers: 42.
- Routed experts per remaining MoE layer: 216.
- Experts per token: 8 routed experts.
- Release status: research candidate, not a validated production legal model.

## Links

- Hugging Face target: https://huggingface.co/ProprietaryLegal/Qwen3.5-91b-a10b-LegalReap-Layerdrop6
- PLI Labs GitHub: https://github.com/ProprietaryLegal/pli-labs
- Model card: [MODEL_CARD.md](MODEL_CARD.md)
- Upload verification: Hugging Face file presence and byte sizes verified on
  2026-06-25; a separate checksum audit is not yet published.
