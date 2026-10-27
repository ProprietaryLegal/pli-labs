# DeepSeek-V4.1-Flash on 2× DGX Spark — status, full source, and what we learned about measuring (2026-10-27)

PLI Labs serves DeepSeek-V4.1-Flash on two DGX Spark (GB10) nodes with tensor parallelism, and offloads two jobs to an
x86 host: the DSpark speculative drafter runs on two NVLinked RTX 3090s, and the model's Engram tables are served from x86
RAM over RDMA. Today we are publishing the full source, plus two focused repositories.

- **Full source:** [ProprietaryLegal/dsv41-offload](https://github.com/ProprietaryLegal/dsv41-offload) — every component,
  with `production` (what serves today), `main` (integration) and a `lane/*` branch per experiment.
- **Measurement harness:** [ProprietaryLegal/dsv41-window-harness](https://github.com/ProprietaryLegal/dsv41-window-harness).
- **Sampled drafting (PP4):** [ProprietaryLegal/dsv41-sampled-drafting](https://github.com/ProprietaryLegal/dsv41-sampled-drafting).

Code only: no weights, prompts, captured requests, generated text or benchmark outputs.

## Where it stands

Labels: MEASURED on our hardware; ESTIMATED by arithmetic; UPSTREAM someone else's number (self-reported, not reproduced).

- Single-stream legal-prose decode 53–57 tok/s and ~2,170 tok/s prefill at 150K tokens (MEASURED). Public two-Spark
  V4.1-Flash builds report 56.8–59.3 tok/s prose decode (UPSTREAM), so for one user we are at the public frontier.
- The remote drafter is the largest single gain: 56.0 vs 41.9 tok/s short and 64.6 vs 41.0 at 32K, speculation on vs off
  (MEASURED). Moving it off the Sparks did not make rounds faster; it freed memory.
- Packed KV pages hold 2,502,600 tokens (+62 % over unpacked) with decode within 1 % (MEASURED).
- Remote Engram is byte-exact and cuts time to first token by 5.4–5.7 % (MEASURED).
- A firmware change on the head node cut the display memory reservation from 2 GB to 512 MiB and returned 1,532 MiB of
  usable memory (MEASURED). We also checked a reported slow memory loss across repeated restarts: across ~30 restarts our
  head node's memory in use changed by 67 MiB, so we do not reproduce it.

## What we learned about measuring

The most useful lessons of the week were about the measurements, not the levers:

1. **An A/A comparison can "lose".** A drafting-depth schedule was rejected on single-user cells — where it is identical to
   the control. The difference was chance. Measure where the arms actually differ.
2. **Make losing as hard as winning.** Counting every secondary metric as a possible loss rejected a no-effect candidate
   16–33 % of the time while accepting it under 2 % (simulation). Secondary metrics now count as a loss only when their
   whole 95 % interval is below zero.
3. **Exact greedy identity is a restore test, not a drafting test.** BF16 near-ties flip at 0.5/1.0-nat logit steps when
   the verify batch changes shape. We use a registered near-tie tolerance — and found a bug in our own checker that refused
   any answer pair of different lengths before looking for the first differing token.
4. **Pair the noise.** At temperature 1, most run-to-run spread is drafter-acceptance luck. Next we give both arms the same
   per-request seeds and many short requests.

An outside review agreed the remaining big wins are a restart-surviving KV cache on the offload host, confidence-gated
drafting priced by the experts each verified draft touches, and a per-token decode time budget. Those are next.

## Credits

Built on DeepSeek's model and DSpark drafter, the vLLM project, HeNryous/deepseek-v41-1017-dgx-spark, sudoingX's
dgx-spark-deepseek-v41 recipe, turboderp's exllamav3 and TensorFold's gather. Licences and notices are in each repository.
