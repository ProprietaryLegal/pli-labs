# B70 Stack Package Summary

## Purpose

The private B70 stack package was built to make local inference repeatable:
kernel/runtime choice, Intel driver/runtime readiness, telemetry, a tested
container lane, and model-format notes.

The public repository does not publish the raw package. It publishes the
lessons that are useful without exposing local paths, hostnames, endpoints,
install logs, checksums, or private service details.

## Public Components

| Component | Public purpose |
| --- | --- |
| Runtime decision | Ensure B70-class devices are visible and stable |
| Driver and firmware lane | Provide the accelerator runtime and telemetry |
| Telemetry tooling | Confirm discovery, health, and memory use |
| XPU API-serving research | Stage the tensor-parallel research lane |
| GGUF-compatible SYCL server | Provide the proven reliability baseline |
| OpenVINO GenAI | Provide an exported-model alternative-stack lane |
| Service-window method | Run candidates, capture public-safe bands, and restore known-good service |

## Lessons

- Separate CUDA and XPU assumptions.
- Pin runtime stacks for benchmark claims.
- Record served model identity before attributing a number.
- Treat exact container tags and runtime versions as internal benchmark
  provenance unless needed for a reproducibility release.
- Do not publish raw launcher files when they contain local service details.
- Do not promote a configuration unless restore and health checks pass.

## Release Boundary

The following are intentionally excluded from the public repo:

- raw logs;
- raw shell launchers;
- service URLs;
- local filesystem paths;
- hostnames;
- model cache locations;
- private environment files;
- credentials;
- client materials;
- binary installers and container archives.
