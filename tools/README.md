# Legal AI Tools

This directory is reserved for public PLI Labs tooling.

PLI tooling should make legal AI more verifiable, not more mysterious. The
preferred pattern is a harness: split legal work into narrow steps, preserve
source provenance, write structured intermediate outputs, and require review
checkpoints before any court-facing or client-facing use.

## Tooling Principles

- Keep source documents and generated claims traceable.
- Prefer deterministic formatting and manifests over ad hoc outputs.
- Make missing source support, schema, or confidence loud.
- Design for local or on-premises operation where confidentiality matters.
- Keep public tools free of private client material and secrets.

## Public Tool README Checklist

Every tool README should explain:

- the legal workflow it supports;
- inputs and outputs;
- where the lawyer review step occurs;
- what the tool refuses to do;
- how to run a smoke test;
- privacy and data-retention expectations.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
