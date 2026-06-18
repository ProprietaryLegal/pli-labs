# Community Research And Adversarial Review

## External Research Pattern

The public V100 ecosystem is fragmented. Useful information comes from:

- V100-specific vLLM forks;
- Volta-compatible attention forks;
- GGUF and `llama.cpp` benchmark reports;
- LMDeploy/TurboMind quantization notes;
- issue threads documenting V100 failures in modern inference stacks;
- community reports from users still operating V100 fleets.

The recurring theme is that V100 can still work well, but only when the stack
explicitly accounts for sm_70.

## Red-Team Conclusions

The research was adversarially reviewed for overclaiming. The red-team findings
were:

- topology guidance is strong, but exact large-model placement must be tested;
- PP x TP for large AWQ MoE models is promising, not proven;
- MTP speedups are plausible but need exact-lane acceptance and wall-clock data;
- CUDA graph removal of eager mode is not a free win on V100;
- backend-specific attention rules should be allowlisted per model;
- configured endpoints are not benchmark evidence unless the server was live and
  identity-checked during the run.

## Public Confidence Levels

| Claim | Public confidence |
|---|---|
| V100 requires float16-first thinking | High |
| Cross-board tensor parallelism is risky | High |
| V100-specific 4-bit kernels matter | High |
| MTP may be a major win | Medium until exact-lane tests |
| PP x TP can serve larger models | Medium until exact-lane load tests |
| CUDA graphs improve V100 lanes | Low-to-medium until measured |

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
