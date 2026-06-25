# Legal REAPs

This directory explains PLI Labs' legal REAP work.

REAP is expert pruning for mixture-of-experts language models. The method
observes which routed experts activate on a calibration workload, ranks experts
by saliency, removes low-saliency experts, and saves a smaller checkpoint.

For legal use, the important question is not simply "how much smaller did the
model get?" The question is: what legal capability survived the cut?

## Why PLI Labs Works On Legal REAPs

Large open MoE models are often too big for small firms or local legal teams to
run comfortably. Legal REAPs are a way to ask whether a model can be reduced
around legal workloads instead of around coding workloads.

The intended benefits are:

- smaller local checkpoints;
- higher context or faster serving on owned hardware;
- easier follow-on finetuning;
- public, inspectable methods for legal-domain model compression.

## Current Public Case Studies

- [Qwen3.5-104b-a10b-LegalReap](Qwen3.5-104b-a10b-LegalReap/README.md)
  ([model card](Qwen3.5-104b-a10b-LegalReap/MODEL_CARD.md),
  [Hugging Face](https://huggingface.co/ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap))
- [Qwen3.5-91b-a10b-LegalReap-Layerdrop6](Qwen3.5-91b-a10b-LegalReap-Layerdrop6/README.md)
  ([model card](Qwen3.5-91b-a10b-LegalReap-Layerdrop6/MODEL_CARD.md),
  [Hugging Face](https://huggingface.co/ProprietaryLegal/Qwen3.5-91b-a10b-LegalReap-Layerdrop6))
- [Qwen3.5 legal REAP research packet](../research/qwen35-legal-reap/README.md)
  and [campaign narrative](../wiki/qwen35-legal-reap-campaign.md)
- [MiniMax-M2.7 legal REAP 0.35](minimax-m2.7-legal-reap-0.35/README.md)
  ([model card](minimax-m2.7-legal-reap-0.35/MODEL_CARD.md),
  [Hugging Face](https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.35))
- [MiniMax-M2.7 legal REAP 0.45](minimax-m2.7-legal-reap-0.45/README.md)
  ([model card](minimax-m2.7-legal-reap-0.45/MODEL_CARD.md),
  [Hugging Face](https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45))
- [MiniMax-M2.7 legal REAP 0.35 GGUF](https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.35-gguf)
- [MiniMax-M2.7 legal REAP 0.45 GGUF](https://huggingface.co/ProprietaryLegal/minimax-m2.7-legal-reap-0.45-gguf)
- [MiniMax-M2.7 REAP 172B-A10B Q4_K_M GGUF](https://huggingface.co/ProprietaryLegal/minimax-m2.7-reap-172b-a10b-q4-k-m-gguf)
- [MiniMax-M2.7 campaign narrative](../wiki/minimax-m2.7-legal-reap-campaign.md)

These MiniMax pages are research case studies. They explain the principles and
the run history. Current Opus-parity bench testing indicates both prunes are
performing well, so they are framed as benchmark-backed candidates while
sanitized score summaries are pending. The linked model cards formalize the
candidate status, intended use, limitations, and privacy boundary without
inventing public benchmark numbers.

The Qwen3.5 pages document a conservative expert-reduced legal REAP and a
Layerdrop6 companion. The Hugging Face repositories are public release locations
for model artifacts and model-card documentation. They intentionally avoid
private infrastructure details such as hostnames, local paths, endpoint ports,
exact deployment topology, and failure runbooks.

## Legal REAP Checklist

Every future validated legal REAP should ship with:

- base model identity and license notes;
- calibration workload summary;
- prune ratio and remaining-expert count;
- model-size and serving notes;
- privacy-safe evaluation set description;
- validation status for calibration, observe diagnostics, and legal eval checks;
- known limitations;
- links to the PLI website.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
