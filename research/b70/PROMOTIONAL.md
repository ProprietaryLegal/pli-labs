# Promotional Material

## Short Positioning

PLI Labs is publishing practical B70 research for local legal AI: what works on
four Intel Arc Pro B70 cards, what fails, and how to benchmark a non-CUDA
inference island without pretending load success is production readiness.

## Longer Positioning

The B70 is compelling because it changes the economics of local AI. Four
32 GB cards create a 128 GB island that can serve a 122B-class reasoning MoE
locally, but only when the software stack is honest about non-CUDA limits.
This research documents the difference between the proven reliability lane
and the still-moving tensor-parallel lane.

PLI's view is simple: local legal AI should be measurable. A credible release
should explain the hardware, the backend, the quantization, the context length,
the benchmark prompt shape, the failures, and the privacy boundary.

## Public Claims

- Four Arc Pro B70 cards can serve a 122B-class GGUF MoE locally through
  llama.cpp SYCL layer-split.
- The proven lane is a reliability result, not a tensor-parallel speed record.
- Long-context legal prompts are a better promotion gate than short synthetic
  decode tests.
- B70 is a serious non-CUDA research lane for private legal AI.
- Intel vLLM / llm-scaler and OpenVINO remain important research paths, but
  each must be judged against measured stability and model-format support.
- Public benchmark claims should separate prompt throughput, decode throughput,
  client output throughput, and total wall throughput.

## Suggested Post Copy

PLI Labs published its B70 research notes: a sanitized stack report, four-card
122B serving results, legal-context benchmark tables, and guardrails for XPU,
SYCL, vLLM, llm-scaler, and OpenVINO lanes.

The short version: B70 is real local AI hardware. The reliable path today is
llama.cpp SYCL layer-split across four 32 GB cards. The research path is still
tensor parallel XPU serving. The discipline is to publish both the wins and
the failure modes.

## Website Copy

PLI Labs tests local inference hardware against legal workloads, not toy
prompts. Our B70 work shows how a non-CUDA 128 GB GPU island can support
large-model legal drafting and review while preserving privacy, auditability,
and cost control.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
