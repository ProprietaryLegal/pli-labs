# Intel vLLM / llm-scaler XPU Findings

## Research Thesis

Intel vLLM and llm-scaler remain important B70-class research paths for future
speed. A mature tensor-parallel XPU stack could become a compute-parallel lane
instead of only a capacity lane.

## What The Public Record Supports

The local research staged and tested XPU lanes, including:

- Intel vLLM / XPU canaries;
- smaller-model legal-context service tests;
- llm-scaler research and staging;
- tensor-parallel configuration research;
- XPU quantization compatibility checks.

The reliable promoted public baseline remains GGUF-compatible SYCL layer-split
serving.

## Public Evidence

The XPU lane is useful for smaller-model service and stack validation. Public
results should be reported as workload tiers and rounded throughput bands, not
raw token counts or launch recipes.

That result matters because it proves the XPU serving lane can be real. It
does not prove large-model tensor parallelism.

## Promotion Rule

A future XPU tensor-parallel lane should be promoted only after it shows:

- intended model identity through the API;
- no worker-init or collective-communication instability;
- successful legal-context gates;
- stable restore or restart behavior;
- rounded prompt, decode, and client-output metrics;
- a measured quality comparison against the current public baseline.

## Public Release Conclusion

The B70 stack story is stronger when it is honest: GGUF-compatible SYCL serving
is the proven public baseline today; Intel vLLM / llm-scaler is the high-upside
research lane for future tensor-parallel speed.
