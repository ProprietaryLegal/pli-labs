# PLI Labs Release Principles

Every public PLI Labs release should explain the reason for the artifact before
it explains the commands.

## 1. Say What Legal Work The Artifact Is For

Legal AI is not one task. A release should say whether it targets drafting,
summarization, evidence review, financial tracing, retrieval, tool selection,
long-context review, refusal behavior, or another workflow.

The release should also say what it is not for. A legal model that is useful for
source-grounded drafting may still be unsafe for unsupervised legal research,
case citation, or final court filing.

## 2. Explain Why Local Matters

PLI materials emphasize two reasons for local or on-premises legal AI:

- confidentiality, when documents should not leave the lawyer's controlled
  environment;
- economics, when repeated legal workflows can be run on owned hardware instead
  of metered cloud calls.

Readmes should connect technical choices to those reasons. Smaller local models
matter because they can be run, audited, and evaluated by the people responsible
for the work.

## 3. Explain The Training Or Calibration Signal

For a finetune, say what instruction behavior the model was trained to improve.
For a REAP, say what calibration workload was used to measure expert saliency.
For a tool, say what source material it expects and what output it produces.

Avoid vague claims such as "optimized for legal." Name the legal capabilities
the artifact tries to preserve.

## 4. Separate Marketing Claims From Verification

A release can be promotional, but it must not blur status. Use clear labels:

- `validated release` - passed the stated validation checks;
- `candidate` - prepared for evaluation but not yet shipped;
- `benchmark-backed candidate` - performing well in current testing, with
  sanitized score details still pending;
- `research case study` - useful method history, not necessarily a final
  release.

MiniMax-M2.7 0.35 and 0.45 currently belong in the benchmark-backed candidate
category because they are performing well in Opus-parity bench testing. Their
public READMEs should be upgraded with sanitized benchmark summaries before
they become formal release cards.

## 5. Preserve Privacy By Design

Public docs must not include client names, private matter details, privileged
communications, credentials, private machine paths, or raw work product. Use
role and workflow descriptions instead of examples copied from live matters.

## 6. Link Back To The PLI Website

Every release README should include:

- https://proprietarylegal.ai
- https://proprietarylegal.com
