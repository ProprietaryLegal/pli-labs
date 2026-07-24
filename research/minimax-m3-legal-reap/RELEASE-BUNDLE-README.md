# MiniMax-M3 Legal REAP 0.22 — Public Documentation Bundle

Public, sanitized documentation for the **MiniMax-M3 Legal REAP 0.22** research
artifact from PLI Labs (Proprietary Legal Intelligence). This bundle is written to
extend the existing `pli-labs` release style (`reaps/`, `huggingface/`,
`research/`); the model weights themselves are distributed through Hugging Face, not
committed here.

## What this is

Most open MoE compression work is calibrated on coding benchmarks. Legal drafting,
long-form analysis, source-grounded summarization, abstention, and judgment-heavy
document work activate different capabilities. This release tests **legal-direction
expert pruning** of MiniMax-M3: a ~428B-parameter, ~23B-active sparse
mixture-of-experts model with 1M-token context and native sparse attention.

The 0.22 artifact is a **text-only, expert-pruned checkpoint** produced by removing
low-saliency routed experts (per-layer adaptive schedule, floor-protected) on the
MoE layers, guided by observations from a private legal-domain calibration corpus.

**It is an unhealed research artifact, under review, with no published evaluations.**
Recovery healing and held-out verification are the next pipeline stages.

## Contents

- [`reaps/minimax-m3-legal-reap-0.22/MODEL_CARD.md`](reaps/minimax-m3-legal-reap-0.22/MODEL_CARD.md)
  — the full model card.
- [`reaps/minimax-m3-legal-reap-0.22/README.md`](reaps/minimax-m3-legal-reap-0.22/README.md)
  — short release note.
- [`huggingface/minimax-m3-legal-reap-0.22/README.md`](huggingface/minimax-m3-legal-reap-0.22/README.md)
  — the Hugging Face model card (mirrors the repo card).
- [`research/minimax-m3-legal-reap/REAP-METHOD.md`](research/minimax-m3-legal-reap/REAP-METHOD.md)
  — what REAP is, and why legal calibration differs from coding calibration.
- [`research/minimax-m3-legal-reap/METHODOLOGY.md`](research/minimax-m3-legal-reap/METHODOLOGY.md)
  — how this artifact was actually produced: two-node streaming observe, sparse
  attention handling at long context, and the streaming split-to-split prune
  serializer.
- [`research/minimax-m3-legal-reap/STREAMING-SERIALIZE-PSEUDOCODE.md`](research/minimax-m3-legal-reap/STREAMING-SERIALIZE-PSEUDOCODE.md)
  — sanitized pseudocode for the model-free prune serializer.
- [`research/minimax-m3-legal-reap/PROVENANCE.md`](research/minimax-m3-legal-reap/PROVENANCE.md)
  — provenance boundary and what the checkpoint's sidecars record.
- [`research/minimax-m3-legal-reap/LIMITATIONS.md`](research/minimax-m3-legal-reap/LIMITATIONS.md)
  — honest limitations of an unhealed legal prune.

## Repository boundaries

- PLI Labs public work is separate from private law-practice automation and client
  work.
- This bundle contains **no** client names, client files, privileged materials,
  private prompts, calibration row contents, hostnames, IP addresses, local file
  paths, credentials, or tokens.
- Model weights are distributed through Hugging Face; only documentation,
  method, and links live here.

## Links

- Website: https://proprietarylegal.com
- Research site: https://proprietarylegal.ai
- REAP method: https://arxiv.org/abs/2510.13999
- MiniMax-M3 base: https://huggingface.co/MiniMaxAI/MiniMax-M3
