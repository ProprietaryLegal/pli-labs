# Current V100 Stack Findings

## Hardware Class

The public research covers multi-GPU V100-class systems with heterogeneous
interconnects. The exact fleet shape, topology, hostnames, and machine-specific
inventory remain internal.

The important lesson is stable: do not treat aggregate VRAM as one uniform
memory pool. Interconnect class, backend behavior, and model architecture
determine whether a configuration is a capacity lane, a speed lane, or merely a
loading experiment.

## Volta Rules

- V100 is a Volta-generation accelerator.
- Public numbers from newer accelerator families should be treated as
  hypotheses until validated on the target V100-class lane.
- Modern low-precision kernels and attention implementations often require
  backend-specific fallbacks.
- GGUF-compatible runtimes remain useful because they provide broad model
  coverage and simple local testing.
- API-serving stacks are useful when they are proven on the exact model family
  and context tier.

## Public Validation Rule

A V100 result should be public only when it has been reduced to:

- broad hardware class;
- backend family;
- model family or parameter class;
- quantization family;
- context tier;
- success/failure category;
- rounded throughput band where measured;
- measured versus inferred status.

Exact topology, failure logs, local paths, endpoints, and private machine
details stay internal. Curated anonymous hardware profiles and successful
settings may be published when they are useful to outside builders without
exposing operational runbooks.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
