# Model Card: Qwen3.5-91b-a10b-LegalReap-Layerdrop6

## Summary

Qwen3.5-91b-a10b-LegalReap-Layerdrop6 is a second Qwen3.5 legal-compression
candidate from PLI Labs. It starts from the expert-reduced REAP-0.16 checkpoint and removes six
decoder layers selected by a calibration-driven layer-importance pass.

The candidate is intentionally framed as an evaluation artifact. It is optimistic because it
opens a practical route toward smaller owned-hardware legal models; it is conservative because
it does not claim production legal reliability before served, identity-verified legal evals are
published.

## Model Details

| Field | Value |
| --- | --- |
| Release name | Qwen3.5-91b-a10b-LegalReap-Layerdrop6 |
| Publisher | PLI Labs / Proprietary Legal Intelligence |
| Release status | Research candidate |
| Base model | Qwen/Qwen3.5-122B-A10B |
| Parent candidate | Qwen3.5-104b-a10b-LegalReap |
| Base license | Apache-2.0 |
| Format | Transformers safetensors checkpoint |
| Compression method | REAP expert reduction plus whole-layer drop |
| Layers | 42 |
| Original layers | 48 |
| Dropped layer indices | 8, 9, 12, 13, 16, 17 |
| Hidden size | 3072 |
| Routed experts | 216 per remaining MoE layer |
| Experts per token | 8 routed experts |
| Indexed weight entries | 27,855 |
| Indexed tensor bytes | 182,327,694,336 |
| Estimated total parameters | about 91B |
| Local artifact size | about 170G |
| Contact | nick@proprietarylegal.com |

## Intended Use

This candidate is intended for:

- controlled legal-domain model compression research;
- owned-hardware serving experiments;
- lawyer-supervised legal drafting and review evaluation;
- comparison against the REAP-0.16 expert-reduced checkpoint and the base model;
- testing whether depth reduction compounds acceptably with conservative expert pruning.

## Out-of-Scope Use

Do not use this candidate for:

- unsupervised legal advice;
- final court filings without attorney review;
- citation generation without independent source verification;
- production deployment based only on this model card;
- claims that layerdrop has been validated for all legal tasks.

## Method

The layerdrop run measured layer importance over legal calibration sequences and removed six
decoder layers with low measured contribution under that pass. The dropped layers were:

`[8, 9, 12, 13, 16, 17]`

The model remains a Qwen3.5 MoE text checkpoint with the same tokenizer and hidden size. The
primary structural change is depth: 48 layers become 42 layers. The routed expert count remains
216 per remaining layer from the parent REAP-0.16 candidate.

## Evaluation Status

Status: research candidate.

The checkpoint is structurally complete and suitable for evaluation. Public aggregate legal
benchmark tables are not yet attached. Until those tables are published, treat the model as a
candidate for research and internal evaluation rather than a validated production legal model.

## Privacy Boundary

This public card intentionally excludes client names, matter names, private source titles, raw
calibration prompts, attorney-client communications, privileged work product, private paths,
credentials, and tokens.

## Limitations

- Depth reduction can affect long-context retention and tool-sensitive reasoning.
- The model may remain fluent while losing source fidelity.
- Public aggregate eval tables are pending.
- This is not an expert-merged checkpoint; it is REAP-0.16 plus layerdrop.
- All legal use requires attorney review and source verification.

## Checksums

Public model weights are distributed through Hugging Face, not committed to
GitHub. Checksums will be added after public weight upload verification.

## Links

- Hugging Face target: https://huggingface.co/ProprietaryLegal/Qwen3.5-91b-a10b-LegalReap-Layerdrop6
- GitHub documentation: https://github.com/ProprietaryLegal/pli-labs
- Proprietary Legal: https://proprietarylegal.com
