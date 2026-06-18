# Known Failures and Guardrails

## Why Publish Failures

Public infrastructure research is more useful when it names failure modes. A
future engineer should not have to rediscover the same unsafe or unproductive
experiments.

## Guardrail Table

| Area | Finding | Public action |
|---|---|---|
| llama.cpp row split | Row/tensor split crashed in userspace on the tested B70 SYCL stack | Keep layer split as the proven path until a newer build is separately gated |
| vLLM tensor parallel | Four-card TP experiments were not production-safe in this evidence set | Treat as guarded research, not a default |
| Continuous batching as a speed fix | Aggregate throughput plateaued around the same decode band for the 122B layer-split lane | Do not expect concurrency alone to turn layer-split into TP |
| Oversized batch settings | Larger batch/microbatch values can win at lower context and fail readiness at larger context | Promote only profiles that pass the target context |
| q8 KV for L2 | q8 K/V was a hard throughput regression in the long legal prompt gate | Keep f16 K/V for the current 122B profile |
| Flash attention off | Disabling flash attention failed readiness in the L2 profile | Keep the validated setting |
| OpenVINO small-model results | OpenVINO HETERO works for supported exported models | Do not compare a smaller export to the 122B lane as if quality were equivalent |
| Exact prompt caching | Exact repeated prompts can be dramatically faster | Do not assume arbitrary follow-up prompts reuse cache without a prompt-format test |

## Failure Classification

The public record separates:

- readiness failures, where an endpoint never became benchmarkable;
- userspace crashes, where the host stayed healthy but the candidate failed;
- host-risk failures, where the experiment can disrupt the machine;
- quality-scope caveats, where a faster smaller model is not a replacement for
  a larger legal drafting model.

## Public Rule

The B70 island should be tuned with service windows and restore checks. A
configuration is not production-safe because it is exciting; it is
production-safe because it survives the legal workload and leaves the system in
a known-good state afterward.
