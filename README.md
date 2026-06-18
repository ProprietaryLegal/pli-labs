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
- [finetunes/README.md](finetunes/README.md) - how PLI thinks about legal
  finetunes after pruning.
- [tools/README.md](tools/README.md) - public tooling principles for legal AI
  pipelines.

## Repository Boundaries

- PLI Labs is distinct from private law-practice automation and client work.
- Public-release work should be separable from private client materials,
  attorney-client communications, and privileged work product.
- Do not add private client files, closed-client materials, credentials, tokens,
  or machine-local secrets to this repo.

## Current Status

Initial public documentation scaffold. The MiniMax-M2.7 0.35 and 0.45 legal
REAP materials are published as benchmark-backed candidate notes. Current
Opus-parity bench testing indicates both prunes are performing well; sanitized
score summaries should be added before any formal model-card release.

## Links

- Website: https://proprietarylegal.ai
- Alternate domain: https://proprietarylegal.com
