# Current B70 Stack Findings

## Hardware Class

The public research covers a B70-class multi-card local inference lane. Exact
card count, topology, hostnames, service ports, and local model paths remain
internal.

The important operational lesson is the same as with any multi-accelerator
local AI system: aggregate memory is not the same thing as one uniform fast
memory pool. Backend selection determines whether a configuration is a capacity
lane, a speed lane, or both.

## Promoted Reliability Lane

The public reliability conclusion is that a GGUF-compatible SYCL layer-split
lane successfully served a large MoE model for long-context legal prompts.

Public-safe configuration facts:

- model class: large MoE GGUF;
- backend family: llama.cpp-compatible SYCL serving;
- split strategy: layer-split capacity lane;
- cache policy: conservative, validated settings;
- workload: long-context legal drafting and review prompts.

Exact launch flags, batch sizes, prompt sizes, internal profile names, and
private validation-window details are intentionally omitted.

## What The Result Means

This proves that B70-class local hardware can support a serious large-model
legal drafting and review lane when the runtime is chosen carefully. It does
not claim that every XPU or tensor-parallel stack is production-ready.

## Public Validation Rule

B70 benchmark promotion should require:

- endpoint readiness, not just process launch;
- a real completion, not just model load;
- legal-workload evidence;
- restore proof after validation-window tests;
- clear distinction between prompt, decode, and client-visible throughput;
- public reporting in rounded bands rather than raw operational rows.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
