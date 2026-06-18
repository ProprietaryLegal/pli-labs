# Successful V100-Class Launch Profiles

This note publishes the useful part of the V100 research record: successful
owned-hardware stacks, settings, and measured throughput. It omits hostnames,
network addresses, local paths, service names, raw failure logs, and operational
runbooks.

## Anonymous Machine Background

- Hardware class: V100-SXM2-32GB multi-GPU Linux workstation.
- Public benchmark profile: nine V100 32GB accelerators, 288 GB aggregate VRAM.
- Host class: high-core-count workstation/server CPU with local model storage.
- Workload: legal drafting, review, and long-context inference probes.
- Measurement status: measured on owned hardware, not inferred from cloud or
  vendor benchmark tables.

The important public lesson is that V100 aggregate memory is useful only after
the model, backend, quantization, context tier, and accelerator grouping are
validated together.

## Stack 1: Ollama / GGUF Local Model Runner

This lane used GGUF-compatible local serving through Ollama-style model
management. The public settings were:

- backend family: Ollama / GGUF local runner;
- quantization family: model-specific GGUF quantization;
- context policy: measured at 8K and at the model's tested maximum context
  where available;
- success gate: the model had to load, accept the legal prompt, and produce a
  completion at the tested context tier.

Measured successful rows:

| Model class | Approx. VRAM | 8K decode tok/s | Max-context decode tok/s | 8K prompt tok/s | Tested max context | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen 30B-A3B-class MoE | 18 GB | 97 | 94 | 325 | 262K | Passed |
| Gemma 26B-class MoE | 17 GB | 88 | 86 | 372 | 262K | Passed |
| Qwen 35B-A3B-class MoE | 24 GB | 62 | 63 | 257 | 262K | Passed |
| Command R 32B-class | 18 GB | 39 | 38 | 501 | 262K | Passed |
| Qwen 122B-A10B-class MoE | 81 GB | 36 | 37 | 112 | 262K | Passed |
| Vision 32B-class | 20 GB | 23 | n/a | 192 | 256K | Passed |
| Dense 31B-class | 19 GB | 24 | 28 | 72 | 262K | Passed |
| Command R+ 104B-class | 59 GB | 12 | 12 | 145 | 65K | Passed |
| Dense 70B reasoning class | 75 GB | 10 | 10 | 90 | 262K | Passed |

The practical conclusion is that medium MoE models were the strongest
interactive V100 lane, while larger MoE and dense reasoning models remained
usable for quality-focused passes when fit and context behavior were confirmed.

## Stack 2: V100-Focused vLLM API Serving

This lane used a V100-focused vLLM API-serving path for AWQ model artifacts.
The public launch shape was:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <local-awq-model> \
  --quantization awq \
  --kv-cache-dtype fp8_e5m2 \
  --max-model-len <context-tier> \
  --tensor-parallel-size <validated-v100-group-size>
```

The concrete private tensor-parallel group, endpoint, local model path, and
service manager are omitted. The successful public settings are the AWQ4
weight path, FP8 E5M2 KV cache storage, and an explicitly validated context
tier.

Measured successful rows used 535 prompt tokens and 256 completion tokens:

| Model class | Context tier | Wall time | Completion tok/s | Result |
| --- | ---: | ---: | ---: | --- |
| 172B-class MoE AWQ | 92,704 | 19.54 s | 13.10 | Loaded and generated |
| 139B-class MoE AWQ | 131,072 | 17.42 s | 14.70 | Loaded and generated |
| 162B-class MoE AWQ | 131,072 | 17.65 s | 14.50 | Loaded and generated |

## Public Conclusion

V100-class owned hardware remains useful for private legal AI when treated as a
validated deployment target rather than a generic pool of VRAM. GGUF/Ollama
lanes were broad and practical. V100-focused API serving worked for selected
AWQ lanes when KV cache mode and context tiers were selected carefully.
