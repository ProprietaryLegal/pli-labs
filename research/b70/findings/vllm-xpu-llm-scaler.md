# Intel vLLM / llm-scaler XPU Findings

## Research Thesis

Intel vLLM and llm-scaler remain the most important B70 research path for
future speed. A mature tensor-parallel XPU stack could use four B70s as a
compute-parallel island instead of only a capacity lane.

## What The Public Record Supports

The local research staged and tested XPU lanes, including:

- Intel vLLM / XPU one-card canaries;
- one-card 7B legal-context service tests;
- llm-scaler research and staging;
- tensor-parallel configuration research;
- XPU quantization compatibility checks.

The reliable 122B promoted lane, however, is still llama.cpp SYCL layer-split.

## One-Card vLLM Evidence

The one-card XPU lane is useful for smaller-model service and stack validation.
It served real legal-context prompts with a 7B-class model and showed
client-visible output in the low-30 tok/s range on longer legal prompt shapes.

That result matters because it proves the XPU serving lane can be real. It does
not prove four-card 122B tensor parallelism.

## llm-scaler Role

llm-scaler is important because it packages Intel's vLLM-XPU work and the
associated runtime stack. It remains the right research category for GPTQ-Int4,
FP8, and tensor-parallel experiments on B70. The public caution is that
research recommendations are not production promotions until they pass the
same legal-context gates as llama.cpp.

## Promotion Rule

A future XPU tensor-parallel lane should be promoted only after it shows:

- the intended model identity through the API;
- no worker-init or collective-communication wedge;
- successful L0, L1, and L2 legal-context gates;
- stable restore or restart behavior;
- clear prompt, decode, and client output metrics;
- a measured quality comparison against the current 122B lane.

## Public Release Conclusion

The B70 stack story is stronger when it is honest: llama.cpp SYCL is the proven
122B lane today; Intel vLLM / llm-scaler is the high-upside research lane for
tomorrow's true tensor-parallel speed.
