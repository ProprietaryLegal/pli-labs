# Software Stack Findings

## Stack Matrix

| Stack | Public status in this research | Role |
| --- | --- | --- |
| GGUF-compatible SYCL serving | Proven reliability lane | Current public baseline |
| Intel vLLM / llm-scaler XPU | Important tensor-parallel research lane | Future speed research |
| OpenVINO GenAI HETERO | Measured alternative-stack lane for supported exported models | Research lane |
| Legacy XPU lanes | Not a preferred foundation for new work | Avoid for new builds |

## GGUF-Compatible SYCL Serving

The GGUF-compatible SYCL lane is the practical winner in the current public
record because it served a large MoE model on B70-class hardware without
promoting unstable collective-heavy paths.

Its limitation is equally important: layer-split serving is a capacity and
reliability result. It should not be described as tensor-parallel speed.

## Intel vLLM / llm-scaler XPU

Intel vLLM and llm-scaler matter because they are plausible routes to future
tensor-parallel speed on B70-class hardware. The public conclusion is that this
stack remains important, but it should not be described as the promoted large
model lane until it passes the same legal-workload stability gates.

## OpenVINO GenAI

OpenVINO GenAI HETERO ran successfully for supported exported models and is a
real research lane. It should be measured separately from the large GGUF lane
because model size, tokenizer, telemetry, and quality are different.

## Quantization Guidance

- GGUF quantization is the proven format for the current public baseline.
- HF-format quantizations remain a research direction for XPU/API-serving
  stacks when backend support is proven.
- CUDA-serving assumptions should not be blindly imported to XPU.
- Cache choices must be benchmarked on legal-context prompts before promotion.

## Public Rule

Do not publish a backend as "working" merely because it starts. Publish it as
working only after it loads, answers, reports model identity, survives the
target workload, and leaves the system healthy afterward.
