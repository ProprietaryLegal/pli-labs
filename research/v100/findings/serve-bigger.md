# Serving Bigger Models On V100

## Core Rule

Serving a bigger model is not the same as serving a model faster.

Using more V100 cards across weaker interconnects can increase capacity, but
single-request decode speed may not improve. The public rule is:

- tensor parallelism inside a tightly connected accelerator group;
- pipeline parallelism or layer split across weaker links;
- avoid cross-board tensor parallelism unless a benchmark proves it is worth the
  communication cost.

## Why Cross-Board Tensor Parallelism Is Risky

Tensor parallelism performs collective communication during model execution.
When that communication crosses a slow link, the slowest link becomes the limit.
That matters most for single-token decode, where communication cannot be hidden
behind much batch work.

Pipeline parallelism moves activations between stages. That can make a larger
model fit, but a single request may still see pipeline bubbles unless there is
enough concurrency.

## Candidate Class

The interesting public candidate class is 200B-plus open MoE models in AWQ or
similar quantized formats. The recommended public method is:

1. first prove PP x TP on a smaller model;
2. then test the larger candidate with conservative context;
3. publish load success, KV capacity, throughput, and failure modes;
4. avoid claiming the large-model lane is validated until it completes real
   prompts and identity checks.

## Hard Ceiling

Models in the 600B-plus class are not just a VRAM math problem. Even when
quantized weights appear to fit, KV cache, backend kernel support, MoE expert
movement, CPU offload bandwidth, and uneven layer sizes can make the lane
impractical. Treat those as research-roadmap targets, not turn-key V100
deployments.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
