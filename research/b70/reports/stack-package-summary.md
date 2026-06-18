# B70 Stack Package Summary

## Purpose

The private B70 stack package was built to make the four-card inference island
repeatable: kernel choice, Intel driver/runtime, xpu-smi telemetry, a tested
container lane, and model-format readiness notes.

The public repository does not publish the raw package. It publishes the
lessons that are useful without exposing local paths, hostnames, endpoints,
install logs, checksums, or private service details.

## Public Components

| Component | Public purpose |
|---|---|
| Kernel/runtime decision | Ensure the B70 cards are visible and stable under the Intel xe / Level Zero path |
| Intel driver and firmware lane | Provide the GPU runtime needed for B70 serving and telemetry |
| xpu-smi | Confirm device discovery, health, memory use, and power/temperature state |
| llm-scaler / vLLM-XPU container research | Stage the tensor-parallel research lane |
| llama.cpp SYCL server | Provide the proven 122B four-card reliability lane |
| OpenVINO GenAI | Provide an exported-model alternative-stack lane |
| Service-window scripts | Run candidates, capture metrics, and restore the known-good service |

## Lessons

- Separate CUDA and XPU assumptions. B70 is not a CUDA card.
- Pin runtime stacks for benchmark claims.
- Record served model identity before attributing a number.
- Treat exact container tags and runtime versions as part of the benchmark.
- Do not publish raw launcher files when they contain local service details.
- Do not promote a configuration unless restore and health checks pass.

## Public Stack Shape

The public stack shape is:

1. B70 hardware discovery and telemetry through Intel tooling.
2. A known-good llama.cpp SYCL server for the 122B reliability lane.
3. XPU/vLLM and llm-scaler lanes for tensor-parallel research.
4. OpenVINO for supported exported-model experiments.
5. Long-context legal benchmark gates as the promotion standard.

## Release Boundary

The following are intentionally excluded from the public repo:

- raw logs;
- raw shell launchers;
- endpoint URLs;
- local filesystem paths;
- hostnames;
- model cache locations;
- private environment files;
- credentials;
- client materials;
- binary installers and container archives.
