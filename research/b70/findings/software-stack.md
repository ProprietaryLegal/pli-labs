# Software Stack Findings

## Stack Matrix

| Stack | Public status in this research | Role |
|---|---|---|
| llama.cpp SYCL / `server-intel` | Proven four-card 122B reliability lane | Current promoted path |
| Intel vLLM / llm-scaler XPU | Important tensor-parallel research lane; not promoted for the 122B path in this evidence set | Future speed lane |
| OpenVINO GenAI HETERO | Measured alternative-stack lane for supported exported models | Research lane |
| IPEX-LLM-style legacy lanes | Not a preferred foundation for new B70 work | Avoid for new builds |

## llama.cpp SYCL

llama.cpp SYCL is the practical winner in the current public record because it
served a 122B-class GGUF model across all four B70 cards without triggering the
stability failures seen in collective-heavy tensor-parallel experiments.

Its limitation is equally important: layer-split does not make all four cards
compute the same layer in parallel. It makes a large model fit and run.

## Intel vLLM / llm-scaler XPU

Intel vLLM and llm-scaler matter because they are the plausible route to real
tensor-parallel speed on B70. The local research staged and tested XPU lanes,
including one-card canaries and llm-scaler research. The public conclusion is
that this stack remains important, but it should not be described as the
promoted 122B production lane until the exact model and topology pass
stability gates.

## OpenVINO GenAI

OpenVINO GenAI HETERO ran successfully for supported exported models and is a
real research lane. It should be measured separately from the 122B GGUF lane
because model size, tokenizer, telemetry, and quality are different.

## Quantization Guidance

- GGUF Q4_K_M is the proven format for the current llama.cpp lane.
- GPTQ-Int4 and related HF-format quantizations remain the preferred research
  direction for XPU/vLLM-style serving when the backend supports the model.
- AWQ assumptions from CUDA serving should not be blindly imported to XPU.
- KV-cache choices must be benchmarked on long legal prompts. In this evidence
  set, f16 K/V was the validated production choice for the 122B lane.

## Public Rule

Do not publish a backend as "working" merely because it starts. Publish it as
working only after it loads, answers, reports model identity, survives the
target prompt shape, and leaves the system healthy afterward.
