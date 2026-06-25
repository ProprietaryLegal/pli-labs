# Model Card: Qwen3.5-104b-a10b-LegalReap

## Summary

Qwen3.5-104b-a10b-LegalReap is a conservative legal-domain expert-pruned candidate from
PLI Labs. It was built to test whether a very large open mixture-of-experts model can be
reduced around legal drafting, factual synthesis, source-grounded review, refusal behavior, and
workflow-sensitive analysis without cutting beyond the stable expert core.

This is a research/evaluation release, not an unsupervised legal advice system. Legal use
requires attorney supervision and independent source verification.

## Model Details

| Field | Value |
| --- | --- |
| Release name | Qwen3.5-104b-a10b-LegalReap |
| Publisher | PLI Labs / Proprietary Legal Intelligence |
| Release status | Research candidate |
| Base model | Qwen/Qwen3.5-122B-A10B |
| Base license | Apache-2.0 |
| Format | Transformers safetensors checkpoint |
| Pruning method | REAP-style routed-expert pruning |
| Router handling | Renormalized over surviving routed experts |
| Working precision | bf16 |
| Layers | 48 |
| Hidden size | 3072 |
| Routed experts | 216 per MoE layer |
| Original routed experts | 256 per MoE layer |
| Experts per token | 8 routed experts |
| Indexed weight entries | 31,839 |
| Indexed tensor bytes | 207,972,470,784 |
| Estimated total parameters | about 104B |
| Local artifact size | about 194G |
| Contact | nick@proprietarylegal.com |

## Intended Use

This candidate is intended for research and controlled evaluation of legal-domain model
compression. The target workflows are:

- lawyer-supervised legal drafting and revision;
- factual and procedural synthesis;
- source-grounded summarization and review;
- legal workflow/tool-use evaluation;
- conservative refusal or escalation when source support is incomplete;
- private/local evaluation by legal teams.

## Out-of-Scope Use

Do not use this candidate for:

- unsupervised legal advice;
- final court filings without attorney review;
- citation generation without independent source verification;
- client-specific work where the user cannot verify the source record;
- production deployment based only on this model card;
- claims of general coding, chat, or benchmark superiority.

## Calibration And Method

The calibration direction was legal work, not generic chat or coding. Public documentation
intentionally does not identify the private legal items used for calibration or evaluation.

The pruning procedure:

1. Observe routed-expert usage under a legal calibration workload.
2. Rank routed experts by REAP-style saliency.
3. Select the stable low-saliency core rather than forcing a larger prune.
4. Remove 40 routed experts per MoE layer, reducing 256 experts to 216.
5. Renormalize routing over survivors.
6. Save a complete Transformers/safetensors checkpoint.

The main technical finding is as important as the artifact: Qwen3.5-122B-A10B appears
prune-resistant. PLI Labs therefore treats 0.16 as the conservative expert-reduction lane and
uses quantization, evaluation, and possible future expert-merge research as the next levers.

## Evaluation Status

Status: research candidate.

The checkpoint is structurally complete and suitable for evaluation. Public aggregate legal
benchmark tables are not yet attached to this card. Until those tables are published, treat the
model as a candidate for research and internal evaluation rather than a validated production
legal model.

## Privacy Boundary

This public card intentionally excludes:

- client names;
- matter names;
- legal source item titles;
- private calibration prompts;
- attorney-client communications;
- privileged work product;
- local hostnames, private paths, credentials, or tokens.

## Limitations

- Strong fluency is not proof of legal reliability.
- Expert pruning can preserve surface quality while degrading source fidelity or tool judgment.
- Public aggregate evaluation tables are still pending.
- This is not an expert-merged checkpoint; it is an expert-reduced REAP checkpoint.
- All legal outputs require attorney review and source verification.

## Checksums

Public model weights are distributed through Hugging Face, not committed to
GitHub. Checksums will be added after public weight upload verification.

## Links

- Hugging Face target: https://huggingface.co/ProprietaryLegal/Qwen3.5-104b-a10b-LegalReap
- GitHub documentation: https://github.com/ProprietaryLegal/pli-labs
- Proprietary Legal: https://proprietarylegal.com
