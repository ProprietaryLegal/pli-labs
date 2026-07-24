# Streaming Split-to-Split Prune Serializer — Pseudocode

Sanitized pseudocode for the model-free serializer that rewrites a MiniMax-M3
checkpoint into an expert-pruned checkpoint without ever loading the model. This is a
faithful description of the method (see [`METHODOLOGY.md`](METHODOLOGY.md), Stage 3),
not the production source; it omits internal engine imports, fail-loud plumbing, and
provenance/census emission.

## Inputs

- `base_dir` — the base checkpoint directory (safetensors shards + index).
- `out_dir` — output directory for the pruned checkpoint.
- `cut_plan` — per-layer decision: for each MoE layer, the ordered tuple of
  **retained** expert ids (and, implicitly, which are dropped). Carries the set of
  dense/skip layers and a stable `cut_plan_hash`.
- `shard_bytes` — target output shard size (e.g. 5 GiB).

## Key transform

```
# Retained expert ids -> dense renumbering, per layer.
renumber[layer] = { old_id: new_id for new_id, old_id in enumerate(retained[layer]) }

def target_key(raw_key):
    # Drop the multimodal stack entirely (text-only output).
    if raw_key starts with ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp."):
        return DROP
    # VL text-stack -> text-only layout (same remap the text-only load performs).
    if raw_key starts with "language_model.model.":
        return "model." + strip_prefix(raw_key, "language_model.model.")
    if raw_key starts with "language_model.lm_head":
        return "lm_head" + strip_prefix(raw_key, "language_model.lm_head")
    if raw_key starts with ("model.", "lm_head."):
        return raw_key                      # already text-only
    if ".layers." in raw_key:
        FAIL_LOUD("decoder tensor with unrecognized prefix")  # never silently drop
    return DROP                             # non-text global tensor
```

## Main loop (streaming, one shard at a time)

```
weight_map   = load_index(base_dir)                 # tensor_key -> source shard file
keys_by_shard = group weight_map keys by shard
writer = GreedyShardWriter(out_dir, shard_bytes,
                           metadata={"format": "pt",
                                     "reap_cut_plan_hash": cut_plan.cut_plan_hash})

for shard in sorted(keys_by_shard):                 # process shards in order
    open shard read-only, on CPU
    for raw_key in sorted(keys_by_shard[shard]):
        key = target_key(raw_key)
        if key is DROP:
            continue

        # 1. Routed expert tensor: model.layers.L.block_sparse_moe.experts.E.w{1,2,3}.weight
        if key matches EXPERT pattern (layer L, expert E, proj P):
            assert L not in cut_plan.skip_layers        # dense layers carry no experts
            assert L in retained                        # planned layer
            if E not in renumber[L]:
                continue                                # dropped expert: omit
            new_key = "model.layers.{L}.block_sparse_moe.experts.{renumber[L][E]}.{P}.weight"
            writer.add(new_key, read_tensor(raw_key))   # byte-identical copy, renumbered
            continue

        # 2. Router weight or routing-bias: index_select retained rows (row slice only)
        if key matches GATE or BIAS pattern (layer L):
            assert L in retained
            idx = tensor(retained[L], dtype=long)
            writer.add(key, read_tensor(raw_key).index_select(0, idx).contiguous())
            continue

        # 3. Everything else (dense MLP, shared expert, attention incl. MSA indexer,
        #    layernorms, embeddings, final norm, lm_head): byte-identical copy.
        writer.add(key, read_tensor(raw_key))

n_shards = writer.finalize()                        # flush + rename to model-XXXXX-of-NNNNN
write_index(out_dir, writer.weight_map, writer.total_bytes)
emit_provenance_and_census(out_dir, cut_plan, ...)  # sidecars
```

## Greedy shard writer

```
class GreedyShardWriter:
    add(key, tensor):
        if key already emitted: FAIL_LOUD("duplicate output tensor key")
        if buffer nonempty and buffered_bytes + nbytes(tensor) > shard_bytes:
            flush()                                 # write current buffer as a part file
        buffer[key] = tensor
    finalize():
        flush()
        if no parts: FAIL_LOUD("refusing to finalize an empty checkpoint")
        rename part files -> model-{i:05d}-of-{total:05d}.safetensors
        rewrite weight_map to final shard names
```

## Properties

- **Peak memory ≈ one shard buffer + one tensor.** The full packed representation and
  any per-layer expert stack are never materialized.
- **Retained experts are bit-identical** to the base; only their ids are renumbered
  into dense positions.
- **Router "renorm" is a pure row slice** — no numeric renormalization of gate logits.
- **Fail-loud, never silent:** unknown decoder-layer prefixes, experts on dense
  layers, router tensors for unplanned layers, duplicate keys, and empty output all
  raise rather than pass.
- **Dry-run mode** (not shown) restricts output to the first N prunable layers for a
  fast on-box I/O and transform check that does not produce a loadable model.
