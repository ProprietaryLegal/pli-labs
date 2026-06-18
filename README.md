# pli-labs

Public legal AI projects from Proprietary Legal Intelligence.

PLI Labs publishes legal-specific model work and legal AI tooling for lawyers
who want systems they can inspect, run locally, and evaluate against real legal
workflows.

The core idea is simple: most open model compression and pruning work is aimed
at coding. Legal drafting, long-form analysis, source-grounded summarization,
and judgment-heavy document work activate different capabilities. A legal model
release should explain what legal workload shaped it, what was intentionally
preserved, what was allowed to degrade, and what evidence supports the result.

Start here:

- [wiki/README.md](wiki/README.md) - public PLI Labs wiki and release thesis.
- [reaps/README.md](reaps/README.md) - what a legal REAP is and why it differs
  from coding-oriented REAPs.
- [huggingface/README.md](huggingface/README.md) - public Hugging Face release
  documentation and model-repository links.
- [finetunes/README.md](finetunes/README.md) - how PLI thinks about legal
  finetunes after pruning.
- [tools/README.md](tools/README.md) - public tooling principles for legal AI
  pipelines.
- [research/v100/README.md](research/v100/README.md) - V100 inference research,
  public-safe benchmark summaries, and testing standards.
- [research/b70/README.md](research/b70/README.md) - Intel Arc Pro B70
  inference research, sanitized stack findings, and legal-context benchmark
  bands.

## Repository Boundaries

- PLI Labs is distinct from private law-practice automation and client work.
- Public-release work should be separable from private client materials,
  attorney-client communications, and privileged work product.
- Do not add private client files, closed-client materials, credentials, tokens,
  or machine-local secrets to this repo.

## Current Status

The MiniMax-M2.7 0.35 and 0.45 legal REAP materials are published as
benchmark-backed candidate notes with Hugging Face release documentation.
Current internal legal evaluation indicates both prunes are promising;
sanitized aggregate score summaries should be added before broader
validated-release claims.

## Links

- Website: https://proprietarylegal.com
- Research site: https://proprietarylegal.ai
