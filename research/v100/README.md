# V100 Research

Public V100 research from Proprietary Legal Intelligence.

This directory publishes public-safe conclusions from PLI Labs' V100-class
owned-hardware research. It covers model serving, backend selection,
benchmarking discipline, and practical deployment limits without exposing
private topology or operational runbooks.

## What Is Published Here

- [ds4-flash-volta-20260707.md](ds4-flash-volta-20260707.md)
  - **DeepSeek-V4-Flash on Volta — full research notes**: first known
  V4-Flash-on-V100 datapoint (12.2 t/s gen / 209 t/s prefill@8K on 8 cards),
  the silent q8_0 K-cache corruption for MLA-576 with the controlled matrix
  that pinned it, the published fail-loud guard (`ds4-volta-fix` branch),
  failed kernel fixes as negative results, KV-cache guidance, the 8-GPU
  peer-mapping ceiling, and full reproduction commands.

- [serving-optimization-2026-07/README.md](serving-optimization-2026-07/README.md)
  - **July 2026 serving-optimization sprint case study**: verified
  model-by-model before/after matrix (122B MoE: measured 39.7 t/s stock
  Ollama baseline → 61.8 t/s llama.cpp / 67.6 t/s vLLM), 36.7–86.6x
  prefix-cache TTFT wins, 2.56–4.5x CUDA-graph decode on sm_70 (incl. the
  first known TP4 graph capture), OOM→served GDN-hybrid patch, ~10x per-GPU
  Gemma lane, KV-quant-at-depth guidance, and published negative results —
  including an honest single-card dense regression from our own tuned build.

- [MASTER-PLAN.md](MASTER-PLAN.md) - public action plan for V100-class serving.
- [findings/current-stack.md](findings/current-stack.md) - public constraints
  for V100-class inference.
- [findings/software-stack.md](findings/software-stack.md) - viable backend
  families and broad failure categories.
- [findings/serve-bigger.md](findings/serve-bigger.md) - how to think about
  models larger than one local accelerator group.
- [findings/successful-launch-profiles.md](findings/successful-launch-profiles.md)
  - successful V100-class launch settings and measured throughput.
- [findings/speedups.md](findings/speedups.md) - speed experiments worth
  testing.
- [findings/community-and-adversarial-review.md](findings/community-and-adversarial-review.md)
  - external-source synthesis plus red-team conclusions.
- [testing/benchmark-summary.md](testing/benchmark-summary.md) - public-safe
  benchmark bands and test conclusions.
- [testing/vllm-v100-probe-results.jsonl](testing/vllm-v100-probe-results.jsonl)
  - coarse public probe rows.
- [reports/](reports/) - sanitized public summaries of earlier V100-class
  benchmark and inference-guide work.
- [reports/device-enumeration-reliability.md](reports/device-enumeration-reliability.md)
  - public-safe device discovery reliability note.
- [PROMOTIONAL.md](PROMOTIONAL.md) - public copy blocks for PLI posts,
  READMEs, and infrastructure positioning.

## Privacy Boundary

The private raw captures included process lists, local paths, hostnames,
network addresses, service inventories, and large search captures. Those are
not published verbatim. The public files publish engineering conclusions and
benchmark bands without private operational details.

No client names, client files, privileged work product, passwords, private
keys, tokens, network addresses, or local filesystem paths should be added
here.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
