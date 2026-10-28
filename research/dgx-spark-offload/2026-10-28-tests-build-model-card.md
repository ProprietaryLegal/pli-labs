# DeepSeek-V4.1-Flash on 2× DGX Spark — republished with tests, build record and a full model card (2026-10-28)

We republished the DSV41 code, refreshed to the evening of 2026-10-27 (the measurement rule's third revision, the
acceptance split) and with five new branches. New with it:

- **Tests for every published tree.** Each branch was tested exactly as published, on CPU, with no network. Every failure
  is classified: it also fails on the private source commit (it needs a GPU or a serving host), it passes when its file
  runs alone, or it needs data that stays outside the release. On `main`: 2,559 passed, 714 failed or errored; 339 of those also fail on the private source commit, 14 pass when run alone, 353 need data on our hosts and 8 need a withheld prompt set or other unpublished data.
  See [`reports/tests/TESTS.md`](https://github.com/ProprietaryLegal/dsv41-offload/blob/main/reports/tests/TESTS.md).
- **A build record for production.** Image digest, pinned commits, serving settings, overlays and the measured
  performance table:
  [`reports/build/BUILD.md`](https://github.com/ProprietaryLegal/dsv41-offload/blob/production/reports/build/BUILD.md).
- **A full model card** for the weights we serve,
  [DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark):
  method, KLD on both panels, measured numbers on our production stack, evaluation status and limitations.

## Production, measured

149 control runs from 11 measurement windows on 2026-10-26/27 (production commit `596ba1c`, GPU clock 2,150 MHz,
512-token answers; medians of selected cells; MEASURED):

| | Decode (tok/s per request) | Time to first token |
|---|---:|---:|
| short prompt, 1 user | 57.5 | 0.8 s |
| 32K prompt, 1 user | 51.8 | 14.5 s |
| 150K prompt, 1 user | 51.7 | 69.2 s |
| 32K prompt, 3 users | 30.2 | 51.5 s |
| 32K prompt, 4 users | 24.8 | 64.6 s |

## New branches in dsv41-offload

- `lane/census` — the expert census: which routed experts and how many bytes each verify round touches, with window
  runners for the next measurements. It prices confidence-gated drafting.
- `lane/matter-cache` — MC1, an opt-in prefix KV tier that survives restarts, bounded per namespace, with a byte-equality
  gate. Code only.
- `lane/lsuite` — the legal evaluation suite: 46 tasks from public opinions and filings, rule-based scoring and a
  pre-registered pass rule. Built, not yet run.
- `lane/dw1` and `lane/dp5` — measurements of the drafter's wait: per-stage step timers, one CUDA graph per step (saves
  0.11–0.12 ms), and a locked-clock sweep for the drafter GPUs.
- `main` now carries DP-2, the acceptance split: acceptance depends mostly on the prompt (1.59 accepted drafts per round
  on one 32K summary prompt over 70 runs, 1.00 on one short drafting prompt over 77), so arms are compared per prompt.

## Corrections

- Sampled drafting (PP4): the offline gain is **+10.2 % accepted drafts** (2.092 → 2.306) and **+6.9 % committed tokens
  per round** (3.092 → 3.306). We had published +10.5 % and +7.1 %, which rounded before dividing. It is still not
  measured live.

## How the release was cleaned

- Host names, user names, addresses and our directory paths are replaced by role names and neutral paths (`sparkhead`,
  `sparkworker`, `x86host`, `controller`; `/opt/dsv41/...`). Paths stay absolute, so path checks behave as before; where
  a changed file's sha256 is pinned elsewhere, the pin was updated.
- Prompt text moved out of the benchmark drivers into files the drivers read at run time (`DSV41_BENCH_SPEC_FILE`,
  `DSV41_LOGIT_AB_PROMPTS_FILE`, `DSV41_PARITY_TEXT_FILE`); the code paths are unchanged.
- On GitHub, repositories whose earlier history held superseded wording were deleted and recreated under the same
  names. On Hugging Face the code repositories were updated in place and their history squashed to a single commit;
  Hugging Face keeps earlier revisions until its own clean-up, so they may stay reachable by commit id for a while.
  Earlier commits of our fork of HeNryous/deepseek-v41-1017-dgx-spark may likewise stay reachable through the upstream
  repository.

## Credits

Built on DeepSeek's model and DSpark drafter, the vLLM project, HeNryous/deepseek-v41-1017-dgx-spark, sudoingX's
dgx-spark-deepseek-v41 recipe, turboderp's exllamav3 and TensorFold's gather. Licences and notices are in each repository.
