# Ollama/GGUF V100 Benchmark Report - April 2026

## Setup

- Hardware class: 9 x Tesla V100-SXM2-32GB.
- Aggregate VRAM: 288 GB.
- Host class: Threadripper PRO workstation.
- Primary serving path tested: Ollama / GGUF.

## Executive Summary

The test showed that several medium-size MoE and dense models run well on V100
hardware at long context. It also showed that some very large modern model
architectures can load but fail at inference time because the backend path
expects GPU features V100 does not provide.

## Leaderboard

| Model family | VRAM class | Gen 8K | Gen max | Prompt 8K | Max context | Notes |
|---|---:|---:|---:|---:|---:|---|
| Qwen 30B-A3B class | 18 GB | 97 tok/s | 94 tok/s | 325 tok/s | 262K | speed leader |
| Gemma 26B MoE class | 17 GB | 88 tok/s | 86 tok/s | 372 tok/s | 262K | speed plus features |
| Qwen 35B-A3B class | 24 GB | 62 tok/s | 63 tok/s | 257 tok/s | 262K | efficient MoE |
| Command R 32B class | 18 GB | 39 tok/s | 38 tok/s | 501 tok/s | 262K | RAG/tool lane |
| Qwen 122B-A10B class | 81 GB | 36 tok/s | 37 tok/s | 112 tok/s | 262K | largest strong working lane |
| Vision 32B class | 20 GB | 23 tok/s | tested | 192 tok/s | 256K | vision lane |
| Dense 31B class | 19 GB | 24 tok/s | 28 tok/s | 72 tok/s | 262K | dense multimodal |
| Command R+ 104B class | 59 GB | 12 tok/s | 12 tok/s | 145 tok/s | 65K | grounded generation |
| Dense 70B reasoning class | 75 GB | 10 tok/s | 10 tok/s | 90 tok/s | 262K | slow reasoning lane |

## Architecture Failures

| Model class | Observed failure | Public lesson |
|---|---|---|
| Large MLA MoE class | loaded, then crashed on first decode | backend path expected unsupported V100 GPU features |
| Large sparse-attention MoE class | loaded, then crashed similarly | loading is not validation |
| Dense 253B class | model loaded but KV allocation failed | uneven layer sizes and KV memory can break apparent fit |

## Recommendations

- Use the fastest medium MoE models for triage, classification, and screening.
- Use larger 100B-class lanes when quality is more important than speed.
- Use vision-specialist lanes for document images and OCR-adjacent tasks.
- Treat very large model load success as only the first gate; decode and KV
  tests are required before calling a lane usable.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
