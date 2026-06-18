# V100 Master Plan

## Thesis

V100-class hardware remains useful for private local AI when the software stack
is chosen around Volta-era limits. The goal is not to claim old hardware beats
new accelerators. The goal is to show how legal teams can evaluate owned
hardware honestly and privately.

## Public Constraints

- Treat newer-GPU performance claims as hypotheses until validated on V100.
- Validate backend family, quantization family, context tier, and model class
  together.
- Distinguish model-load success from usable generation.
- Publish rounded result bands rather than raw probe logs.
- Keep topology, exact device counts, private launch commands, and failure
  runbooks internal.

## Current Best Direction

Use three public lanes:

- GGUF-compatible runtimes for broad compatibility and fallback testing.
- API-serving runtimes where V100 support is proven for the target model class.
- Specialized dense-model lanes where the Volta path is explicitly validated.

## Public Validation Standard

Each public benchmark should publish:

- broad hardware class;
- model family or parameter class;
- backend family;
- quantization family;
- context tier;
- rounded throughput band;
- success/failure category;
- measured, inferred, or recommendation status.

## PLI Positioning

PLI's V100 work is infrastructure research for private, local legal AI. The
commercial point is practical: legal teams need systems they can inspect, run
locally, and evaluate against real workflows without exposing client work to
unnecessary third-party infrastructure.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
