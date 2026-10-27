# pli-labs

Public legal AI projects from Proprietary Legal Intelligence.

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

PLI Labs publishes legal-specific model work and legal AI tooling for lawyers
who want systems they can inspect, run locally, and evaluate against real legal
workflows.

The core idea is simple: most open model compression and pruning work is aimed
at coding. Legal drafting, long-form analysis, source-grounded summarization,
and judgment-heavy document work activate different capabilities. A legal model
release should explain what legal workload shaped it, what was intentionally
preserved, what was allowed to degrade, and what evidence supports the result.

Start here:

- [wiki/README.md](wiki/README.md) - public PLI Labs wiki and release thesis.
- [reaps/README.md](reaps/README.md) - what a legal REAP is and why it differs
  from coding-oriented REAPs.
- [huggingface/README.md](huggingface/README.md) - public Hugging Face release
  documentation and model-repository links.
- [research/qwen35-legal-reap/README.md](research/qwen35-legal-reap/README.md)
  - Qwen3.5 legal REAP research packet, public settings, and artifact
  inventory.
- [blog/proprietarylegal/README.md](blog/proprietarylegal/README.md) - public
  blog prose for ProprietaryLegal.com.
- [finetunes/README.md](finetunes/README.md) - how PLI thinks about legal
  finetunes after pruning.
- [tools/README.md](tools/README.md) - public tooling principles for legal AI
  pipelines.
- [research/v100/README.md](research/v100/README.md) - V100 inference research,
  public-safe benchmark summaries, and testing standards.
- [research/b70/README.md](research/b70/README.md) - Intel Arc Pro B70
  inference research, sanitized stack findings, and legal-context benchmark
  bands.
- [Swift 1.5 on 4x V100](https://github.com/ProprietaryLegal/swift-1.5-v100) - 1Cat-vLLM 1.5.1 serving recipe for
  Swift 1.5 (a Qwen3.8-27B fine-tune) on four V100-SXM2-32GB, and a blind A/B against Qwen3.8-27B: same quality on six
  legal-drafting tasks at about half the reasoning ([Hugging Face](https://huggingface.co/ProprietaryLegal/swift-1.5-27b-v100-serving)).
- [DeepSeek-V4.1-Flash on 2× DGX Spark](research/dgx-spark-offload/README.md) - x86 Engram and RTX 3090
  drafter offload, async handoff, long-context tools, and exact acceptance-ceiling analysis.
  [Full-stack source](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
  [Hugging Face collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e). Standalone repositories are canonical; hub folders are snapshots.
  - [Async host-flag handoff](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff) · [HF](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff).
  - [Remote RTX 3090 drafter](https://github.com/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) · [HF](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090).
  - [NVLink two-card drafter](https://github.com/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) · [HF](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink).
  - [Remote Engram](https://github.com/ProprietaryLegal/dsv41-remote-engram-rdma) · [HF](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma).
  - [ofl RDMA/TCP transport](https://github.com/ProprietaryLegal/ofl-rdma-transport) · [HF](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport).
  - [Native drafter on SM86](https://github.com/ProprietaryLegal/dsv41-native-1017-drafter-sm86) · [HF](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86).
  - [Long-context benchmark tools](https://github.com/ProprietaryLegal/dsv41-longctx-bench-tools) · [HF](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools).
  - [Exact acceptance ceiling](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling) · [HF](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling).
  - [Full stack](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · [HF](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).
  - [Offload findings](https://github.com/ProprietaryLegal/dsv41-spark-offload-findings) · [HF](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings).
  - **2026-10-27:** [full source (all branches)](https://github.com/ProprietaryLegal/dsv41-offload) · [measurement harness](https://github.com/ProprietaryLegal/dsv41-window-harness) · [sampled drafting (PP4)](https://github.com/ProprietaryLegal/dsv41-sampled-drafting) · [status and lessons](research/dgx-spark-offload/2026-10-27-status-and-code.md).

## Repository Boundaries

- PLI Labs is distinct from private law-practice automation and client work.
- Public-release work should be separable from private client materials,
  attorney-client communications, and privileged work product.
- Do not add private client records, closed-client materials, credentials, tokens,
  or machine-local secrets to this repo.

## Current Status

The MiniMax-M2.7 0.35 and 0.45 legal REAP materials are published as
benchmark-backed candidate notes with Hugging Face release documentation.

The Qwen3.5-104b-a10b-LegalReap and
Qwen3.5-91b-a10b-LegalReap-Layerdrop6 artifacts are public research
candidates derived from Qwen/Qwen3.5-122B-A10B. The expert-reduced checkpoint
is about 104B parameters; the Layerdrop6 companion is about 91B parameters.
Public aggregate legal benchmark tables should be added before broader
validated-release claims.

## Links

- Website: https://proprietarylegal.com
- Research site: https://proprietarylegal.ai

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
