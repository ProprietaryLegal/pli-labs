# Ollama/GGUF V100 Benchmark Report - Public Summary

## Setup

The benchmark covered a V100-class local workstation using GGUF-compatible
serving paths. Exact device count, aggregate memory, host class, and raw probe
rows are retained internally.

## Executive Summary

The test showed that several medium-size MoE and dense models run well on
V100-class hardware at long context. It also showed that some very large modern
architectures can load but fail when the backend path expects accelerator
features the hardware does not provide.

## Public Result Bands

| Model family | Public result band | Public conclusion |
| --- | --- | --- |
| Medium MoE | high local throughput | Strong interactive lane |
| Mid-size dense or multimodal | moderate to high local throughput | Useful specialist lane |
| Large MoE | moderate local throughput | Useful when quality matters and fit is proven |
| Large dense reasoning | lower local throughput | Quality-focused lane, not speed lane |
| Very large modern architecture | mixed readiness | Loading is not validation |

## Recommendations

- Use medium MoE models for triage, classification, and screening.
- Use larger lanes when quality is more important than speed.
- Use specialist lanes for document images and OCR-adjacent tasks.
- Treat load success as only the first gate; decode and cache tests are
  required before calling a lane usable.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
