# Legal REAP Thesis

REAP-style expert pruning is attractive for legal AI because many large open
models are mixture-of-experts models. Only a subset of the experts activate for
any particular workload. If a legal calibration set reliably uses some experts
and barely uses others, a pruned checkpoint can be smaller while preserving the
capabilities the legal workload needs.

## Why Legal REAPs Differ From Coding REAPs

Most public pruning and local-model optimization work is built around coding or
general chat. That is useful, but it can point the knife at the wrong tissue for
legal work.

A coding-oriented REAP may preserve code synthesis, tool-call patterns, and
programming-language reasoning while stripping out experts that carry legal
writing, formal prose, long factual synthesis, careful refusal behavior, or
source-grounded advocacy. From a coding benchmark, that looks efficient. For a
lawyer, it may be the exact wrong trade.

A legal REAP starts from a different premise: preserve legal writing,
long-document synthesis, evidence-sensitive reasoning, and lawyer-verifiable
output, even if unrelated coding ability or casual chat quality degrades.

## What A Legal REAP Must Explain

A public legal REAP README should identify:

- the base model and architecture;
- the legal calibration workload;
- the target prune ratio and how many experts remain;
- whether router weights were renormalized after pruning;
- what capabilities were intended to survive;
- what capabilities were acceptable to lose;
- what validation evidence supports the current status;
- what the artifact should and should not be used for.

## Validation Matters

PLI's internal REAP work produced a hard lesson: surface fluency is not enough.
Router renormalization can make a damaged model look confident. A public release
should therefore separate:

- calibration realization: did the intended legal workload actually reach the
  observer;
- observe diagnostics: did critical layers and capabilities survive the cut;
- strengthened evaluation: does the pruned model hold up against the base model
  on legal tasks that include tool selection, retrieval, long context, and
  reasoning.

If later evidence changes the status of a prune, the README should say so
plainly. A good release history can distinguish candidates, validated releases,
and method notes without hiding uncertainty.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
