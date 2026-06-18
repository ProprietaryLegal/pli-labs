# B70 Master Plan

## Thesis

The Intel Arc Pro B70 is an unusually interesting local-AI card because it
pairs 32 GB of VRAM with a non-CUDA software stack. Four cards create a
128 GB inference island that can run a 122B-class MoE model locally when the
serving path respects the actual backend limits.

The opportunity is not "B70 replaces every NVIDIA lane." The opportunity is a
separate, inspectable, non-CUDA legal AI lane that can run large private
workloads, expose OpenAI-compatible APIs, and provide a cost-aware alternative
to cloud inference.

## Hard Constraints

- B70 serving is a Level Zero / SYCL / XPU problem, not a CUDA problem.
- Aggregate VRAM is not automatically a fast shared memory pool.
- Layer-split serving can make a model fit across four cards, but it does not
create tensor-parallel compute.
- Tensor-parallel stacks promise more throughput, but they are only production
usable after stability is proven on the exact kernel, driver, runtime, and
model combination.
- Short synthetic decode tests are only canaries. Legal work needs long
prefill, longer answers, restore proof, and output sanity.
- Raw stack packages often contain private paths, endpoint details, and machine
state. Public releases should publish conclusions, not unfiltered logs.

## Current Best Direction

Use three lanes:

- Dockerized llama.cpp `server-intel` for the proven 122B GGUF reliability
  lane across four B70s.
- Intel vLLM / llm-scaler XPU lanes for smaller one-card service and future
  tensor-parallel research.
- OpenVINO GenAI HETERO for exported-model experiments where the model format
  and runtime support are already proven.

## Highest-Value Experiments

1. Keep the four-card llama.cpp SYCL lane as the public reliability baseline.
2. Publish long-context legal benchmarks with prompt tokens, generated tokens,
   context size, batch settings, and separate prompt/decode/client metrics.
3. Continue XPU/vLLM work first on one-card and small-model lanes, then promote
   multi-card tensor parallelism only after stability is demonstrated.
4. Use OpenVINO HETERO for supported exported models, but do not compare a
   small exported model to a 122B model as if quality were equivalent.
5. Treat row-split, unstable TP collectives, and oversized batch settings as
   research items until they pass service-window gates.

## Public Validation Standard

Each public benchmark should publish:

- model family and parameter class;
- backend and version family;
- GPU count and topology class;
- quantization and KV-cache mode;
- context length;
- prompt tokens and generated tokens;
- prompt throughput, decode throughput, client output throughput, and total
  wall throughput where available;
- load success or readiness failure;
- whether the number is measured, inferred, or only a recommendation.

## PLI Positioning

PLI's B70 work is infrastructure research for private legal AI. The commercial
point is practical: legal teams need systems that can be inspected, run
locally, and evaluated against real drafting and review workloads. B70 makes
that question interesting because it expands local serving capacity outside the
usual CUDA path.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
