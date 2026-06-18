# V100 Public Benchmark Summary

## April 2026 Ollama/GGUF Benchmarks

The April benchmark tested a 9 x V100-SXM2-32GB system with GGUF/Ollama lanes.
The key public result was that medium MoE models were the speed leaders, while
some very large modern architectures loaded but failed during inference on V100.

| Model family | VRAM class | Generation speed | Max context tested | Result |
|---|---:|---:|---:|---|
| Qwen 30B-A3B class | ~18 GB | 94-100 tok/s | 262K | fastest tested lane |
| Gemma 26B MoE class | ~17 GB | 86-91 tok/s | 262K | best speed/features balance |
| Qwen 35B-A3B class | ~24 GB | 62-63 tok/s | 262K | strong single-GPU MoE lane |
| Command R 32B class | ~18 GB | 38-39 tok/s | 262K | useful RAG/structured lane |
| Qwen 122B-A10B class | ~81 GB | 35-40 tok/s | 262K | largest smart working lane in that run |
| Vision 32B class | ~20 GB | 23 tok/s | 256K | vision specialist lane |
| Dense 31B class | ~19 GB | 24-28 tok/s | 262K | dense multimodal lane |
| Command R+ 104B class | ~59 GB | ~12 tok/s | 65K | grounded generation lane |
| Dense 70B reasoning class | ~75 GB | ~10 tok/s | 262K | slow but high-quality reasoning lane |
| DeepSeek large MLA class | ~229 GB loaded | crash | n/a | V100 architecture incompatibility |
| GLM large sparse-attention class | ~263 GB loaded | crash | n/a | V100 architecture incompatibility |
| Dense 253B class | ~141 GB loaded | KV failure | n/a | uneven memory/KV issue |

## June 2026 vLLM V100 Probe

The June probe tested large MoE AWQ and GGUF serving behavior through a V100
vLLM lane. Public conclusions:

- fp8_e5m2 KV storage made the successful large AWQ lanes fit.
- A 172B-class AWQ lane loaded at 92,704 context and generated about
  13.10 completion tok/s.
- 139B/162B-class AWQ lanes loaded at 131,072 context and generated about
  14.5-14.7 completion tok/s.
- The same 172B-class AWQ lane failed at 131,072 context due to insufficient KV
  cache capacity.
- GGUF through that vLLM path failed because the architecture was unsupported in
  that backend.
- fp16 KV at 131,072 context failed for the tested 172B-class lane because the
  available KV memory was too small.

The compact public JSONL is in
[vllm-v100-probe-results.jsonl](vllm-v100-probe-results.jsonl).

## What Counts As Public Evidence

Numbers in this folder are public evidence only when they include:

- model family;
- backend class;
- quantization and KV mode;
- context length;
- load success or failure;
- generated tokens and wall-clock throughput for successful runs;
- failure reason for failed runs.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
