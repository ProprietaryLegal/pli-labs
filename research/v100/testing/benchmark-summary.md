# V100 Public Benchmark Summary

## Public-Safe Summary

PLI Labs' V100-class testing shows that older enterprise accelerators remain
useful for local legal AI when the model, backend, quantization path, and
context tier are validated together.

The public benchmark record is summarized in bands for broad comparisons.
Curated successful launch rows with exact measured token rates are published in
[successful-launch-profiles.md](../findings/successful-launch-profiles.md).
Endpoint settings, failure thresholds, local paths, and private capacity
details remain internal.

## Observed Result Bands

| Model class | Backend family | Context tier | Public result band | Conclusion |
| --- | --- | --- | --- | --- |
| Medium MoE | GGUF-compatible | long | high local throughput | Strong local lane for interactive work |
| Large MoE | GGUF-compatible | long | moderate throughput | Useful when fit and cache behavior are proven |
| Large MoE | API-serving stack | long | moderate throughput | Viable only after backend-specific validation |
| Dense reasoning | GGUF-compatible | long | lower throughput | Potentially useful for quality-focused passes |
| Very large modern architectures | mixed | long | mixed or failed readiness | Loading is not proof of usable inference |

## Public Evidence Standard

Future public V100 benchmark notes should publish:

- model family, not private model path;
- backend family, not endpoint details;
- broad context tier, not exact private prompt shape;
- rounded throughput band, not raw timing rows;
- success, readiness failure, or incompatibility category;
- measured versus inferred status.

## Links

- Coarse public rows: [vllm-v100-probe-results.jsonl](vllm-v100-probe-results.jsonl)
- Successful launch profiles:
  [successful-launch-profiles.md](../findings/successful-launch-profiles.md)
- https://proprietarylegal.com
- https://proprietarylegal.ai
