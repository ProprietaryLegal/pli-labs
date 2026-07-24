# MiniMax-M3 Legal REAP 0.22

A text-only, expert-pruned research derivative of `MiniMaxAI/MiniMax-M3`, calibrated
on legal work rather than coding benchmarks.

- **Method:** REAP-style routed-expert pruning at an overall 0.22 ratio, using a
  floor-enforced per-layer adaptive schedule. Retained routed experts range 89–111
  of 128 per MoE layer; dense layers 0–2 and the shared expert are preserved; top-4
  routing is unchanged. The vision tower is dropped (text-only).
- **Size:** ~330B total parameters after prune (base ~428B), ~23B active.
- **Status:** unhealed raw prune, under review, **no published evaluation**.
  Recovery healing and held-out KL / faithfulness verification are the next stages.
- **Use:** research and lawyer-supervised evaluation only. Not a legal assistant;
  not for production or unsupervised legal advice.

See [`MODEL_CARD.md`](MODEL_CARD.md) for the full card and
[`../../research/minimax-m3-legal-reap/`](../../research/minimax-m3-legal-reap/) for
the method.

Weights: https://huggingface.co/ProprietaryLegal/minimax-m3-legal-reap-0.22

Built with MiniMax M3. Derivative of MiniMax-M3 under the MiniMax Community License.
All legal use requires attorney supervision and independent source verification.
