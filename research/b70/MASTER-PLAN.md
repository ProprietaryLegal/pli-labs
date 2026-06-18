# B70 Master Plan

## Thesis

B70-class hardware is interesting for local legal AI because it expands the
owned-hardware serving conversation beyond CUDA. The opportunity is a private,
inspectable local lane for legal drafting and review workloads, not a claim
that one hardware class replaces every other accelerator.

## Public Constraints

- Treat aggregate memory as capacity until backend behavior proves otherwise.
- Distinguish layer-split capacity from tensor-parallel speed claims.
- Validate on legal workloads, not only short synthetic prompts.
- Publish result bands and promotion status, not raw launch rows.
- Keep topology, validation-window details, ports, launchers, and failure runbooks
  internal.

## Current Best Direction

Use three public lanes:

- GGUF-compatible SYCL serving for the validated reliability baseline.
- Intel XPU API-serving research for smaller or future tensor-parallel lanes.
- Exported-model runtimes where format support and workload quality are proven.

## Public Validation Standard

Each public benchmark should publish:

- broad hardware class;
- model family or parameter class;
- backend family;
- quantization family;
- context tier;
- rounded throughput band;
- success/readiness category;
- measured, inferred, or recommendation status.

## PLI Positioning

PLI's B70 work is infrastructure research for private legal AI. The commercial
point is practical: legal teams need systems that can be inspected, run locally,
and evaluated against real drafting and review workloads.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
