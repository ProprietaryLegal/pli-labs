# Promotional Material

## Short Positioning

PLI Labs is publishing practical V100 research for local AI: what still
works on Volta, what fails, and how to benchmark old enterprise GPUs without
pretending they are H100s.

## Longer Positioning

Many firms can buy or repurpose older GPU hardware long before they can justify
premium cloud inference or new data-center GPUs. V100s are not modern, but they
are still useful when the model stack respects their limits. This research shows
how to think about V100 serving honestly: float16-first, topology-aware, careful
about quantization kernels, skeptical of unmeasured speedups, and disciplined
about public benchmark evidence.

## Public Claims

- V100 can still run useful local AI workloads.
- The main problem is not raw VRAM; it is choosing a backend that actually
  supports sm_70.
- Medium MoE models can be very fast on V100.
- Large 100B-class lanes are practical when fit, KV cache, and backend support
  are verified.
- Loading a huge model is not proof that inference works.
- Public numbers should distinguish measured V100 results from extrapolated
  claims.

## Suggested Post Copy

PLI Labs published its V100 research notes: benchmark summaries, backend
findings, topology rules, and a public testing standard for old enterprise GPUs.
The short version: V100 is not dead, but it punishes lazy assumptions. Use
float16, verify the quantization kernel, keep tensor parallelism inside fast
links, and publish failures as well as wins.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
