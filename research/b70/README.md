# B70 Research

Public Intel Arc Pro B70 research from Proprietary Legal Intelligence.

This directory publishes public-safe conclusions from PLI Labs' B70-class
owned-hardware research for legal AI. It focuses on the practical difference
between "a model loads" and "a model is useful for legal drafting and review."

The public record intentionally avoids exact topology, private launch commands,
endpoint details, internal file paths, and failure runbooks.

## What Is Published Here

- [MASTER-PLAN.md](MASTER-PLAN.md) - public plan and validation boundary.
- [findings/current-stack.md](findings/current-stack.md) - sanitized stack
  findings for B70-class local inference.
- [findings/software-stack.md](findings/software-stack.md) - backend choices
  and broad quantization guidance.
- [findings/llamacpp-sycl-success.md](findings/llamacpp-sycl-success.md) -
  public-safe summary of the successful GGUF serving lane.
- [findings/successful-launch-profile.md](findings/successful-launch-profile.md)
  - successful B70-class launch settings and measured legal throughput.
- [findings/vllm-xpu-llm-scaler.md](findings/vllm-xpu-llm-scaler.md) -
  tensor-parallel research status for Intel vLLM and llm-scaler lanes.
- [findings/known-failures-and-guardrails.md](findings/known-failures-and-guardrails.md)
  - public guardrails without exact failed settings.
- [testing/benchmark-summary.md](testing/benchmark-summary.md) - benchmark
  tiers and public promotion conclusions.
- [testing/b70-public-results.jsonl](testing/b70-public-results.jsonl) -
  coarse public benchmark rows.
- [reports/](reports/) - public summaries of the B70 stack package and
  official hardware facts.
- [PROMOTIONAL.md](PROMOTIONAL.md) - public copy blocks for PLI posts,
  READMEs, and infrastructure positioning.

## Hugging Face Link

- B70-friendly GGUF artifact:
  https://huggingface.co/ProprietaryLegal/minimax-m2.7-reap-172b-a10b-q4-k-m-gguf

## Privacy Boundary

The private raw captures included hostnames, local paths, LAN addresses,
service ports, process output, shell launchers, model cache paths, logs,
installer checksums, and operational validation details. Those are not
published verbatim. The public files publish engineering conclusions and
benchmark bands without exposing the private deployment.

No client names, client files, privileged work product, passwords, private
keys, tokens, LAN addresses, or local filesystem paths should be added here.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai
