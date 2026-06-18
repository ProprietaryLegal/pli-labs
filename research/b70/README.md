# B70 Research

Public Intel Arc Pro B70 research from Proprietary Legal Intelligence.

This directory publishes the sanitized research record for running large legal
AI workloads on a 4 x Intel Arc Pro B70 inference island. It focuses on
hardware facts, software stack selection, long-context benchmark discipline,
and the practical gap between "a model loads" and "a model is useful for legal
drafting and review."

## What Is Published Here

- [MASTER-PLAN.md](MASTER-PLAN.md) - the public action plan for B70 serving.
- [findings/current-stack.md](findings/current-stack.md) - hardware and
  production-stack findings.
- [findings/software-stack.md](findings/software-stack.md) - backend choices,
  stack maturity, and quantization guidance.
- [findings/llamacpp-sycl-success.md](findings/llamacpp-sycl-success.md) - the
  proven four-card llama.cpp SYCL result.
- [findings/vllm-xpu-llm-scaler.md](findings/vllm-xpu-llm-scaler.md) -
  tensor-parallel research status for Intel vLLM and llm-scaler lanes.
- [findings/known-failures-and-guardrails.md](findings/known-failures-and-guardrails.md)
  - failures that should not be rediscovered casually.
- [testing/benchmark-summary.md](testing/benchmark-summary.md) - public
  benchmark tables and promotion conclusions.
- [testing/b70-public-results.jsonl](testing/b70-public-results.jsonl) -
  compact sanitized benchmark rows.
- [reports/](reports/) - public summaries of the B70 stack package and
  official hardware facts.
- [PROMOTIONAL.md](PROMOTIONAL.md) - public copy blocks for PLI posts, READMEs,
  and infrastructure positioning.

## Privacy Boundary

The private raw captures included hostnames, local paths, LAN addresses,
endpoint ports, process output, shell launchers, model cache paths, logs,
installer checksums, and operational service-window details. Those are not
published verbatim. The public files publish the engineering conclusions and
benchmark evidence without exposing the private deployment.

No client names, client files, privileged work product, passwords, private
keys, tokens, LAN addresses, or local filesystem paths should be added here.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
