# Known Failures and Guardrails

## Why Publish Guardrails

Public infrastructure research is more useful when it explains what not to
claim. This page keeps the lesson while omitting exact failed settings and
machine-specific recovery details.

## Public Guardrails

| Area | Public finding | Public action |
| --- | --- | --- |
| Split strategy | Some split modes did not pass promotion gates | Keep the validated capacity lane as the public baseline |
| Tensor parallel XPU | Collective-heavy paths need separate stability proof | Treat as research until gated |
| Oversized batching | Larger settings can help one tier and fail another | Promote only settings that pass the target workload |
| Cache policy | Cache changes can trade fit, speed, and stability | Validate on legal-context tiers before promotion |
| Smaller exports | Smaller exported models can be fast | Do not compare them to larger legal models as if quality were equivalent |
| Prompt caching | Repeated prefixes can be faster | Validate real workflow prompts before relying on cache reuse |

## Public Rule

A configuration is not production-safe because it is exciting. It is
production-safe only when it survives the legal workload, returns usable output,
and leaves the system in a known-good state afterward.

Exact failed launch settings, host-risk classifications, recovery steps, and
private validation-window details remain internal.
