# V100 Research

Public V100 research from Proprietary Legal Intelligence.

This directory publishes the sanitized research record for running large language
models on V100-class hardware. It is about V100 inference, model serving,
hardware topology, backend selection, benchmarking, and practical deployment
limits. It is not a client-work repository and does not contain private source
materials.

## What Is Published Here

- [MASTER-PLAN.md](MASTER-PLAN.md) - the public action plan for V100 serving.
- [findings/current-stack.md](findings/current-stack.md) - hardware and backend
  facts that constrain V100 serving.
- [findings/software-stack.md](findings/software-stack.md) - viable backends and
  backend failure modes.
- [findings/serve-bigger.md](findings/serve-bigger.md) - how to think about
  models larger than one V100 board.
- [findings/speedups.md](findings/speedups.md) - speed experiments worth testing.
- [findings/community-and-adversarial-review.md](findings/community-and-adversarial-review.md)
  - external-source synthesis plus red-team conclusions.
- [testing/benchmark-summary.md](testing/benchmark-summary.md) - public benchmark
  tables and test conclusions.
- [testing/vllm-v100-probe-results.jsonl](testing/vllm-v100-probe-results.jsonl)
  - compact sanitized probe results.
- [reports/](reports/) - public summaries of the older 9-GPU benchmark report,
  the V100 inference guide, and the missing-GPU hardware analysis.
- [PROMOTIONAL.md](PROMOTIONAL.md) - public copy blocks for PLI posts, READMEs,
  and model-infrastructure positioning.

## Privacy Boundary

The private raw captures included process lists, local paths, hostnames, LAN
addresses, endpoint inventories, and very large grep outputs. Those are not
published verbatim. The public files publish the technical findings, benchmark
numbers, and recommended experiments without private operational details.

No client names, client files, privileged work product, passwords, private keys,
tokens, LAN addresses, or local filesystem paths should be added here.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
