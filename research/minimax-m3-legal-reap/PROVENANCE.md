# Provenance — MiniMax-M3 Legal REAP 0.22

## What the checkpoint carries

The pruned checkpoint ships with two machine-readable sidecars written at
serialization time:

- **`reap_prune_provenance.json`** — records the base checkpoint index hash, the
  observation tensor path and its sha256, the approved ratio, the selection metric,
  the seed, the full per-layer cut plan (skip layers, prunable layers, per-layer
  retained counts, whether the schedule is non-uniform, a schedule summary), and the
  **cut-plan hash**. It also records the code revisions of the pruning engine used, so
  the exact selection logic is auditable.
- **an expert-count census** — the per-layer retained/dropped expert counts, used to
  cross-check the serialized config against the intended cut plan.

The serializer also stamps `reap_cut_plan_hash` into the safetensors shard metadata,
so the shards themselves are tied to the cut plan.

## Public provenance fields

| Field | Value |
| --- | --- |
| Base model | `MiniMaxAI/MiniMax-M3` |
| Base revision | fill at publish (base checkpoint revision) |
| Derivative | text-only expert-pruned checkpoint |
| Prune ratio | 0.22 (floor-enforced, per-layer adaptive) |
| Selection metric | REAP saliency (canonical `reap` metric) |
| Seed | 42 |
| Retained experts/layer | 89–111 of 128 |
| Dense layers preserved | 0, 1, 2 |
| Shared expert | preserved |
| Cut-plan hash | fill at publish (from `reap_prune_provenance.json`) |
| Shard total size | fill at publish (from the safetensors index) |

## What is deliberately excluded

Provenance is recorded accurately, but the **contents** of the private calibration
corpus are never published: no client names, no matter facts, no source-document
titles, no raw prompts, no calibration row contents, and no private file paths,
hostnames, or IP addresses. The observation sha256 in the sidecar identifies the
observation tensors without revealing their inputs.

## How to reproduce the cut decision

Given the base observation tensors, the profile knobs, the seed (42), and the ratio
(0.22), the per-layer cut plan is a deterministic function: the adaptive schedule fixes
each layer's retained count under capability floors, then per-layer lowest-saliency
selection picks the dropped experts. Recomputing over the same observations must
reproduce the same `cut_plan_hash`. The observations themselves derive from the private
calibration corpus and are not distributed.
