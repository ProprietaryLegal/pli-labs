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

- [MiniMax-M2.7 legal REAP 0.35](minimax-m2.7-legal-reap-0.35/README.md)
  ([model card](minimax-m2.7-legal-reap-0.35/MODEL_CARD.md))
- [MiniMax-M2.7 legal REAP 0.45](minimax-m2.7-legal-reap-0.45/README.md)
  ([model card](minimax-m2.7-legal-reap-0.45/MODEL_CARD.md))
- [MiniMax-M2.7 campaign narrative](../wiki/minimax-m2.7-legal-reap-campaign.md)

These MiniMax pages are research case studies. They explain the principles and
the run history. Current Opus-parity bench testing indicates both prunes are
performing well, so they are framed as benchmark-backed candidates while
sanitized score summaries are pending. The linked model cards formalize the
candidate status, intended use, limitations, and privacy boundary without
inventing public benchmark numbers.

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

- https://proprietarylegal.ai
- https://proprietarylegal.com
