# B70 Public Benchmark Summary

## Public-Safe Summary

PLI Labs' B70-class research confirms that a GGUF-compatible SYCL layer-split
lane can serve a large MoE model for legal drafting and review workloads. The
public benchmark record is summarized by tier and rounded result band.

Private profile names, endpoint details, local paths, and failed timing rows
remain internal. Curated successful settings and measured token rates are
published in
[successful-launch-profile.md](../findings/successful-launch-profile.md).

## Benchmark Tiers

| Tier | Workload shape | Public result band | Conclusion |
| --- | --- | --- | --- |
| Short legal context | legal drafting/review canary | strong local throughput | Promoted as a readiness gate |
| Medium legal context | longer factual/legal synthesis | strong local throughput | Promoted after validation-window checks |
| Long legal context | large record-style prompt | moderate local throughput | Useful with validated prompt format |
| Smaller exported model | supported OpenVINO-style lane | strong throughput | Useful research lane, not a substitute quality claim |

## Cache Finding

Exact repeated prompt prefixes can substantially reduce repeated-context cost.
This is a workflow optimization, not a raw model-speed claim. Realistic
follow-up prompts must preserve the relevant prefix to benefit.

## Public Evidence Standard

Future public B70 benchmark notes should publish:

- broad hardware class;
- model family or parameter class;
- backend family;
- quantization family;
- context tier;
- rounded throughput band;
- success/readiness category;
- measured, inferred, or recommendation status.

## Links

- Coarse public rows: [b70-public-results.jsonl](b70-public-results.jsonl)
- Successful launch profile:
  [successful-launch-profile.md](../findings/successful-launch-profile.md)
- Hugging Face GGUF: https://huggingface.co/ProprietaryLegal/minimax-m2.7-reap-172b-a10b-q4-k-m-gguf
- https://proprietarylegal.com
- https://proprietarylegal.ai
