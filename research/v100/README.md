# V100 Research

Public V100 research from Proprietary Legal Intelligence.

This directory publishes public-safe conclusions from PLI Labs' V100-class
owned-hardware research. It covers model serving, backend selection,
benchmarking discipline, and practical deployment limits without exposing
private topology or operational runbooks.

## What Is Published Here

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
