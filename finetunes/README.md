# Legal Finetunes

This directory is reserved for PLI Labs legal finetune materials.

PLI's public position is that finetuning should be tied to concrete legal
workflows, not generic style transfer. A useful legal finetune should teach a
model how to behave inside a repeatable workflow: read the source packet, follow
the task boundary, produce a lawyer-reviewable draft, and preserve enough
structure for downstream checking.

## Why Finetune After A REAP

A successful legal REAP can make a large MoE model smaller and easier to serve.
That may also make local finetuning more practical. The intended sequence is:

1. choose a base MoE model with legal-writing potential;
2. calibrate and prune around legal workloads;
3. verify that legal capability survives;
4. finetune the smaller checkpoint on task-specific legal behavior;
5. evaluate against held-out legal workflows.

Do not finetune on top of an unbenchmarked prune and call it a release. The
prune must first be good enough to preserve the capabilities the finetune needs,
and the finetune should be evaluated against held-out legal workflow tasks.

## What A Finetune README Should Explain

- the base or pruned checkpoint;
- the legal workflow being trained;
- the training-data shape, without private examples;
- what behavior should improve;
- what behavior is out of scope;
- held-out evaluation design;
- privacy and licensing boundaries.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
